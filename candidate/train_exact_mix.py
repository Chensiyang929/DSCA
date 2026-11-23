"""Exact mainline for one target image and its one-level Internal-Wald pair.

The outer target may be the original Full observation (MS128/PAN512) or one
official RR observation (MS64/PAN256).  In both cases the spatial ratio is 4
and the Internal-Wald target is exactly the outer observation MS.
"""
import os, sys, runpy
from pathlib import Path
import h5py, numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));sys.path.insert(0,str(R/'ZSVD-test-for-mix/zup'))
g=int(os.environ['EXACT_GT']);f=int(os.environ['EXACT_FULL']);assert g>=0 and f>=0 and g+f>0
schedule_mode=os.environ.get('EXACT_SCHEDULE_MODE','interleaved')
assert schedule_mode in {'interleaved','full_then_gt','gt_then_full','random'}
data_path=Path(sys.argv[sys.argv.index('--data_path')+1])
with h5py.File(data_path) as obs:
 assert 'gt' not in obs and set(['ms','lms','pan'])<=set(obs)
 ms_hw=tuple(obs['ms'].shape[-2:]);out_hw=tuple(obs['lms'].shape[-2:])
 assert ms_hw[0]==ms_hw[1] and out_hw==(4*ms_hw[0],4*ms_hw[1])
 assert tuple(obs['pan'].shape[-2:])==out_hw
 assert ms_hw[0]%4==0 and ms_hw[1]%4==0
 observation_ms=np.asarray(obs['ms'][:],np.float32)
if g>0:
 assert '--reduced_data_path' in sys.argv,'GT/Mix requires --reduced_data_path'
 wald_path=Path(sys.argv[sys.argv.index('--reduced_data_path')+1])
 with h5py.File(wald_path) as w:
  inner_ms_hw=(ms_hw[0]//4,ms_hw[1]//4)
  assert tuple(w['ms'].shape[-2:])==inner_ms_hw
  assert tuple(w['pan'].shape[-2:])==tuple(w['lms'].shape[-2:])==tuple(w['gt'].shape[-2:])==ms_hw
  assert np.array_equal(np.asarray(w['gt'][:],np.float32),observation_ms)
else:
 assert '--reduced_data_path' not in sys.argv,'Full-only must not receive --reduced_data_path'
for flag,value in [('--use_reduced','1' if g>0 else '0'),('--reduced_patch_size','0'),('--reduced_steps_per_epoch','1')]:
 if flag in sys.argv:assert sys.argv[sys.argv.index(flag)+1]==value
 else:sys.argv += [flag,value]
import training_ratio_config as cfg
cfg.get_training_cycle=lambda sensor:(g,f)
def exact_schedule(epochs):
 periodic=[i%(g+f)<g for i in range(epochs)]
 gt_total=sum(periodic)
 if schedule_mode=='interleaved':return periodic
 if schedule_mode=='full_then_gt':return [False]*(epochs-gt_total)+[True]*gt_total
 if schedule_mode=='gt_then_full':return [True]*gt_total+[False]*(epochs-gt_total)
 seed=int(os.environ.get('EXPERIMENT_SEED','10'))
 rng=np.random.default_rng([seed,epochs,gt_total])
 order=np.array([True]*gt_total+[False]*(epochs-gt_total))
 rng.shuffle(order)
 return order.tolist()
cfg.epoch_schedule=lambda sensor,epochs,reduced_ratio=None:exact_schedule(epochs)
source_text=(R/'train_SDE_instrumented.py').read_text(encoding='utf-8')
assert '"patch_policy": "whole_image_internal_wald"' in source_text
internal_label=str(ms_hw[0]) if g>0 else 'disabled'
print(f'EXACT_MAINLINE_PASS outer={out_hw[0]} internal_gt={internal_label} cycle={g}:{f} schedule={schedule_mode}',flush=True)
trainer = os.environ.get('EXACT_STUDENT_TRAINER', 'train_pruned_normalized_spatial.py')
if Path(trainer).name != trainer:
    raise ValueError('EXACT_STUDENT_TRAINER must be a filename inside the code directory')
runpy.run_path(str(R/trainer),run_name='__main__')
