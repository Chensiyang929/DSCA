"""Per-scene, post-update HQNR selection using the original DLPan functions.

This is metric-guided adaptation on the Full scene, not loss-based selection
or an independent held-out HQNR test. No optimizer or gradient steps occur here.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
from scipy.io import loadmat, savemat
import torch

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_backend(toolbox_root=None, engine='auto'):
    explicit = toolbox_root or os.environ.get('DLPAN_ROOT')
    if explicit and not (Path(explicit) / 'Tools/indexes_evaluation_FS.m').is_file():
        raise FileNotFoundError(f'Invalid explicit DLPan root: {explicit}')
    candidates = [toolbox_root or os.environ.get('DLPAN_ROOT'), HERE.parent / 'DLPan',
                  HERE.parent.parent / 'DLPan-Toolbox-main' / '02-Test-toolbox-for-traditional-and-DL(Matlab)']
    root = next((Path(p).resolve() for p in candidates if p and
                 (Path(p) / 'Tools/indexes_evaluation_FS.m').is_file()), None)
    if root is None:
        raise FileNotFoundError('Specify --dlpan_root containing Tools and Quality_Indices.')
    for relative in ('Quality_Indices/HQNR.m', 'Quality_Indices/D_lambda_K.m', 'Tools/MTF.m'):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    octave = shutil.which('octave-cli') or shutil.which('octave')
    matlab = shutil.which('matlab')
    if engine in ('auto', 'octave') and octave:
        prefix = [octave]
        return root, prefix + ['--quiet', '--eval']
    if engine in ('auto', 'matlab') and matlab:
        return root, [matlab, '-batch']
    raise RuntimeError('HQNR requires MATLAB or Octave with image/signal packages; no metric fallback is allowed.')


class DLPanHQNR:
    def __init__(self, toolbox_root=None, engine='auto', timeout=300):
        self.root, self.command = resolve_backend(toolbox_root, engine)
        self.timeout = timeout
        self.provenance = {
            'toolbox_root': str(self.root), 'command': self.command,
            'flagQNR': 0, 'ratio': 4, 'thvalues': 0, 'block_size': 32,
            'gf2_mtf': 'Unchanged toolbox genMTF fallback GNyq=0.3; not a calibrated GF2 sensor response',
            'wrapper_sha256': file_sha256(HERE / 'score_epoch_hqnr.m'),
            'source_hashes': {str(p.relative_to(self.root)): file_sha256(p)
                              for folder in ('Tools', 'Quality_Indices')
                              for p in sorted((self.root / folder).rglob('*.m'))},
        }

    def score(self, arrays, sensor):
        sensor = sensor.upper()
        if sensor not in ('WV3', 'WV2', 'QB', 'GF2'):
            raise ValueError(sensor)
        for name in ('proposed', 'I_MS_LR', 'I_MS', 'I_PAN'):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f'Non-finite input: {name}')
        # Only our temporary scoring files are removed, never experiment data.
        with tempfile.TemporaryDirectory(prefix='zs_hqnr_') as tmp:
            source, target = Path(tmp) / 'input.mat', Path(tmp) / 'score.mat'
            savemat(source, {key: arrays[key] for key in ('proposed', 'I_MS_LR', 'I_MS', 'I_PAN')})
            quote = lambda p: "'" + str(p).replace('\\', '/').replace("'", "''") + "'"
            expr = (f'addpath({quote(HERE)}); score_epoch_hqnr('
                    f'{quote(source)},{quote(target)},{quote(self.root)},{quote(sensor)});')
            result = subprocess.run(self.command + [expr], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                    errors='replace', timeout=self.timeout, check=False)
            if result.returncode or not target.is_file():
                raise RuntimeError(f'DLPan HQNR failed ({result.returncode}): {result.stdout[-4000:]}')
            values = loadmat(target)
            score = {name: float(np.asarray(values[name]).item()) for name in ('HQNR', 'Dl', 'Ds')}
            if not all(math.isfinite(v) for v in score.values()):
                raise ValueError(f'Non-finite HQNR result: {score}')
            return score


@contextlib.contextmanager
def evaluation_state(model):
    """Evaluation must not consume patch-sampling RNG or alter train/eval flags."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, flag in flags:
            module.training = flag
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class HQNRSelector:
    def __init__(self, data_path, sensor, scene, model_dir, identifier, epochs,
                 device, backend=None, toolbox_root=None, engine='auto'):
        self.backend = backend or DLPanHQNR(toolbox_root, engine)
        self.sensor, self.scene, self.epochs = sensor.upper(), int(scene), int(epochs)
        self.device, self.directory = torch.device(device), Path(model_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identifier = identifier
        self.history = self.directory / f'{identifier}_hqnr_history.jsonl'
        self.metadata = self.directory / f'{identifier}_hqnr_selection.json'
        self.checkpoint = self.directory / f'{identifier}_best_hqnr.pth'
        if self.history.exists() or self.metadata.exists() or list(self.directory.glob(f'{identifier}_*.pth')):
            raise FileExistsError(f'Use a fresh model directory; existing scene artifacts: {identifier}')
        self.norm = 1023.0 if self.sensor == 'GF2' else 2047.0
        with h5py.File(data_path, 'r') as f:
            ms, lms, pan = [np.array(f[key][scene], dtype=np.float32) / self.norm
                            for key in ('ms', 'lms', 'pan')]
        if pan.ndim == 2:
            pan = pan[None]
        self.lms = torch.from_numpy(lms).unsqueeze(0).to(self.device)
        self.pan = torch.from_numpy(pan).unsqueeze(0).to(self.device)
        self.inputs = {'I_MS_LR': ms.transpose(1, 2, 0) * self.norm,
                       'I_MS': lms.transpose(1, 2, 0) * self.norm,
                       'I_PAN': pan.squeeze(0) * self.norm}
        self.best, self.completed = None, 0
        self.record = dict(sensor=self.sensor, scene=self.scene, epoch_budget=self.epochs,
                           selection_metric='HQNR', selection_data=str(Path(data_path).resolve()),
                           selection_rule='post-update every epoch; strict maximum; earliest tie',
                           inference='FP32 full image; residual + LMS; no clipping; no GT',
                           metric_guided_adaptation=True, evaluator=self.backend.provenance,
                           completed=False)

    def update(self, model, epoch, branch):
        if epoch != self.completed + 1 or epoch > self.epochs:
            raise ValueError('Every epoch must be evaluated once in order.')
        with evaluation_state(model), torch.autocast(device_type=self.device.type, enabled=False):
            fused = (model(self.lms, self.pan) + self.lms).squeeze(0)
            proposed = fused.permute(1, 2, 0).cpu().numpy() * self.norm
            score = self.backend.score(dict(self.inputs, proposed=proposed), self.sensor)
        if not all(math.isfinite(score[key]) for key in ('HQNR', 'Dl', 'Ds')):
            raise ValueError('Non-finite HQNR score; no checkpoint was selected.')
        improved = self.best is None or score['HQNR'] > self.best['HQNR']
        row = dict(epoch=epoch, branch=branch, **score, selected=improved)
        if improved:
            tmp = self.checkpoint.with_suffix('.pth.tmp')
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, self.checkpoint)
            self.best = dict(epoch=epoch, **score)
        with self.history.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        self.completed = epoch
        self.record.update(evaluated_epochs=epoch, best=self.best,
                           checkpoint=self.checkpoint.name, checkpoint_sha256=file_sha256(self.checkpoint))
        atomic_json(self.metadata, self.record)
        return row

    def finish(self):
        if self.completed != self.epochs or self.best is None:
            raise RuntimeError('Cannot complete an incompletely evaluated training run.')
        self.record['completed'] = True
        atomic_json(self.metadata, self.record)
