"""Fixed activation-pruned initialization and split-only GT+teacher recovery."""
import argparse,hashlib,json,sys,time
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'support'))
import h5py,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from teacher.fusionnet import FusionNet as Teacher
from pruned_fusionnet import PrunedFusionNet


class TrainSet(Dataset):
 def __init__(self,p):self.file=h5py.File(p,'r');self.length=len(self.file['gt'])
 def __len__(self):return self.length
 def __getitem__(self,i):return tuple(torch.as_tensor(np.asarray(self.file[k][i],np.float32)/2047.) for k in ['lms','pan','gt'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def copy_pruned(student,teacher,selected):
 s=torch.as_tensor(selected,dtype=torch.long)
 with torch.no_grad():
  student.conv1.weight.copy_(teacher.conv1.weight[s]);student.conv1.bias.copy_(teacher.conv1.bias[s])
  for name in ['res1','res2','res3','res4']:
   a,b=getattr(student,name),getattr(teacher,name)
   for conv in ['conv20','conv21']:
    x,y=getattr(a,conv),getattr(b,conv);x.weight.copy_(y.weight[s][:,s]);x.bias.copy_(y.bias[s])
  student.conv3.weight.copy_(teacher.conv3.weight[:,s]);student.conv3.bias.copy_(teacher.conv3.bias)


def select_channels(teacher,ds,bands):
 loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0);score=torch.zeros(32,device='cuda');seen=0
 with torch.no_grad():
  for lms,pan,_ in loader:
   lms,pan=lms.cuda(),pan.cuda();z=teacher.relu(teacher.conv1(pan.repeat(1,bands,1,1)-lms));score+=z.abs().mean((0,2,3));
   for block in [teacher.res1,teacher.res2,teacher.res3,teacher.res4]:z=block(z);score+=z.abs().mean((0,2,3))
   seen+=len(lms)
   if seen>=128:break
 # Fixed half-width rule; stable channel-index tie break.
 return sorted(range(32),key=lambda i:(-float(score[i]),i))[:16],seen,score.cpu().tolist()


def main():
 p=argparse.ArgumentParser();p.add_argument('--sensor',choices=['WV3','QB'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.manual_seed(10);torch.cuda.manual_seed_all(10);torch.backends.cudnn.deterministic=True;bands=8 if a.sensor=='WV3' else 4;ds=TrainSet(a.data);teacher=Teacher(bands).cuda().eval();state=torch.load(a.teacher,map_location='cuda',weights_only=True);state=state.get('state_dict',state);teacher.load_state_dict({str(k).removeprefix('module.'):v for k,v in state.items()});selected,selection_samples,scores=select_channels(teacher,ds,bands);model=PrunedFusionNet(bands).cuda();copy_pruned(model,teacher,selected);a.output.parent.mkdir(parents=True,exist_ok=True);init=a.output.with_name(a.output.stem+'_sliced_init.pth');torch.save(model.state_dict(),init)
 opt=torch.optim.Adam(model.parameters(),lr=5e-4,betas=(.9,.999));g=torch.Generator().manual_seed(10);dl=DataLoader(ds,batch_size=16,shuffle=True,generator=g,num_workers=0,pin_memory=True);history=[];updates=0;start=time.time()
 for epoch in range(1,51):
  lossrows=[];model.train()
  for lms,pan,gt in dl:
   lms,pan,gt=lms.cuda(non_blocking=True),pan.cuda(non_blocking=True),gt.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);pred=model(lms,pan)+lms
   with torch.no_grad():target=teacher(lms,pan)
   loss=.5*(pred-gt).square().mean()+.5*(pred-target).square().mean();loss.backward();opt.step();updates+=1;lossrows.append(float(loss.detach()))
  history.append(float(np.mean(lossrows)));print(epoch,history[-1],flush=True)
 torch.save(model.state_dict(),a.output);meta=dict(sensor=a.sensor,architecture='PrunedFusionNet width16 full residual RF21',parameters=sum(x.numel() for x in model.parameters()),selection_rule='top16 of 32 by mean absolute teacher features across conv1 and four block outputs; fixed first 128+ samples; tie by channel index',selected_channels=selected,selection_samples=selection_samples,activation_scores=scores,train_file=str(a.data.resolve()),train_sha256=sha(a.data),teacher_file=str(a.teacher.resolve()),teacher_sha256=sha(a.teacher),epochs=50,batch_size=16,lr=5e-4,loss='0.5 GT MSE + 0.5 original FusionNet output MSE',seed=10,checkpoint='last only; no test/validation metric selection',updates=updates,loss_history=history,sliced_init_sha256=sha(init),checkpoint_sha256=sha(a.output),seconds=time.time()-start);a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['loss_history','activation_scores']}))


if __name__=='__main__':main()
