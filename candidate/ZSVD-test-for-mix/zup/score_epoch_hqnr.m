function score_epoch_hqnr(input_path, output_path, toolbox_root, sensor)
% Exact Full-resolution scoring protocol used by assess_search.m.
% No GT, clipping, rounding, additional cropping, or proxy Python metric.
if exist('OCTAVE_VERSION', 'builtin')
    pkg load image;
    pkg load signal;
end
addpath(fullfile(toolbox_root, 'Tools'));
addpath(fullfile(toolbox_root, 'Quality_Indices'));
old = pwd; cleanup = onCleanup(@() cd(old)); cd(toolbox_root);
f = load(input_path);
L = 11; if strcmpi(sensor, 'GF2'), L = 10; end
[Dl, Ds, HQNR] = indexes_evaluation_FS(double(f.proposed), ...
    double(f.I_MS_LR), double(f.I_PAN), L, 0, double(f.I_MS), sensor, 4, 0);
assert(all(isfinite([Dl, Ds, HQNR])));
if exist('OCTAVE_VERSION', 'builtin')
    save('-mat7-binary', output_path, 'Dl', 'Ds', 'HQNR');
else
    save(output_path, 'Dl', 'Ds', 'HQNR', '-v7');
end
end
