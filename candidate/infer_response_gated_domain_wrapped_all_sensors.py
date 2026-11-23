import argparse,sys
from pathlib import Path
import h5py,numpy as np,torch
from scipy.io import savemat
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from response_gated_domain_wrapped_pruned_fusionnet import ResponseGatedDomainWrappedPrunedFusionNet
def main():
 p=argparse.ArgumentParser();p.add_argument('--sensor',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--data_path',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',choices=['base','shared','head'],required=True);p.add_argument('--domain',choices=['gt','full'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();s=a.sensor.upper();b=8 if s in ('WV3','WV2') else 4;n=1023. if s=='GF2' else 2047.
 with h5py.File(a.data_path) as f:l=np.asarray(f['lms'][a.scene],np.float32);pan=np.asarray(f['pan'][a.scene],np.float32);ms=np.asarray(f['ms'][a.scene],np.float32);gt=np.asarray(f['gt'][a.scene],np.float32) if 'gt' in f else None
 d=torch.device(a.device);lt=torch.from_numpy(l/n).unsqueeze(0).to(d);pt=torch.from_numpy(pan/n).unsqueeze(0).to(d);pt=pt.unsqueeze(1) if pt.ndim==3 else pt;m=ResponseGatedDomainWrappedPrunedFusionNet(b,a.variant).to(d);m.load_state_dict(torch.load(a.checkpoint,map_location=d,weights_only=True));m.eval()
 with torch.no_grad():pred=(lt+m(lt,pt,domain=a.domain))[0].cpu().numpy().transpose(1,2,0)*n
 out={'proposed':pred.astype(np.float32),'I_MS':l.transpose(1,2,0),'I_PAN':pan.squeeze(0),'I_MS_LR':ms.transpose(1,2,0)}
 if gt is not None:out['gt']=gt.transpose(1,2,0)
 q=Path(a.output);q.parent.mkdir(parents=True,exist_ok=True);savemat(q,out,do_compression=False)
if __name__=='__main__':main()
