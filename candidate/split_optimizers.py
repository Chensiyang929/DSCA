"""Independent Adam history and explicit learning rates for GT and Full."""
import os
import torch


def make_optimizers(parameters, lr):
    parameters = list(parameters)
    gt_lr = float(os.environ.get('MAINLINE2_GT_LR', lr))
    full_lr = float(os.environ.get('MAINLINE2_FULL_LR', lr))
    print(f'SPLIT_ADAM_LR gt={gt_lr:.12g} full={full_lr:.12g}', flush=True)
    return (
        torch.optim.Adam(parameters, lr=gt_lr, betas=(0.9, 0.999)),
        torch.optim.Adam(parameters, lr=full_lr, betas=(0.9, 0.999)),
    )


def install_split_optimizers(source):
    old = 'optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))'
    assert source.count(old) == 1
    source = source.replace(old, 'from split_optimizers import make_optimizers\noptimizer_gt, optimizer_full = make_optimizers(model_student.parameters(), lr)')
    selection = '        use_reduced_epoch = (reduced_data_loader is not None) and selected_schedule[epoch - 1]'
    assert source.count(selection) == 1
    return source.replace(selection, selection + '\n        optimizer = optimizer_gt if use_reduced_epoch else optimizer_full')
