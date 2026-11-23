"""Two independent Adams with lower backbone LR and normal adapter LR."""
import os,torch
def make_module_optimizers(model,lr,variant):
 gt=float(os.environ.get('MAINLINE2_GT_LR',lr));full=float(os.environ.get('MAINLINE2_FULL_LR',lr));scale=float(os.environ.get('WRAP_BACKBONE_LR_SCALE','0.1'));adapter_scale=float(os.environ.get('WRAP_ADAPTER_LR_SCALE','1.0'));control_scale=float(os.environ.get('WRAP_CONTROL_LR_SCALE','1.0'));backbone=model.backbone_parameters();named=[(n,p) for n,p in model.named_parameters() if not n.startswith('backbone.')];keys=('response_logits','blend_logit','gate_raw','gate_logit','band_gain');controls=[p for n,p in named if any(k in n for k in keys)];adapters=[p for n,p in named if not any(k in n for k in keys)]
 def groups(rate):
  if variant=='base' or not adapters:return [{'params':backbone,'lr':rate}]
  result=[{'params':backbone,'lr':rate*scale}]
  if adapters:result.append({'params':adapters,'lr':rate*adapter_scale})
  if controls:result.append({'params':controls,'lr':rate*control_scale})
  return result
 print(f'MODULE_SPLIT_ADAM variant={variant} gt={gt:.12g} full={full:.12g} backbone_scale={1.0 if variant=="base" else scale:.12g} adapter_scale={0.0 if variant=="base" else adapter_scale:.12g}',flush=True)
 return torch.optim.Adam(groups(gt),betas=(.9,.999)),torch.optim.Adam(groups(full),betas=(.9,.999))
def install_module_optimizers(source):
 old='optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))';assert source.count(old)==1;source=source.replace(old,"from module_split_optimizers import make_module_optimizers\noptimizer_gt, optimizer_full = make_module_optimizers(model_student, lr, os.environ['WRAP_VARIANT'])")
 marker='        use_reduced_epoch = (reduced_data_loader is not None) and selected_schedule[epoch - 1]';assert source.count(marker)==1;return source.replace(marker,marker+'\n        optimizer = optimizer_gt if use_reduced_epoch else optimizer_full')
