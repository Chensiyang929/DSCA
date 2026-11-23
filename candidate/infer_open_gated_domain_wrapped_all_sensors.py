import argparse, sys
from pathlib import Path
import h5py, numpy as np, torch
from scipy.io import savemat
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from open_gated_domain_wrapped_pruned_fusionnet import OpenGatedDomainWrappedPrunedFusionNet


def main():
    p=argparse.ArgumentParser();p.add_argument('--sensor',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--data_path',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',choices=['base','shared','head'],required=True);p.add_argument('--domain',choices=['gt','full'],required=True);p.add_argument('--initial-gate',type=float,default=.95);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    sensor=a.sensor.upper();bands=8 if sensor in ('WV3','WV2') else 4;norm=1023. if sensor=='GF2' else 2047.
    with h5py.File(a.data_path) as f:lms=np.asarray(f['lms'][a.scene],np.float32);pan=np.asarray(f['pan'][a.scene],np.float32);ms=np.asarray(f['ms'][a.scene],np.float32);gt=np.asarray(f['gt'][a.scene],np.float32) if 'gt' in f else None
    device=torch.device(a.device);lt=torch.from_numpy(lms/norm).unsqueeze(0).to(device);pt=torch.from_numpy(pan/norm).unsqueeze(0).to(device);pt=pt.unsqueeze(1) if pt.ndim==3 else pt
    model=OpenGatedDomainWrappedPrunedFusionNet(bands,a.variant,a.initial_gate).to(device);model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True));model.eval()
    with torch.no_grad():pred=(lt+model(lt,pt,domain=a.domain))[0].cpu().numpy().transpose(1,2,0)*norm
    out={'proposed':pred.astype(np.float32),'I_MS':lms.transpose(1,2,0),'I_PAN':pan.squeeze(0),'I_MS_LR':ms.transpose(1,2,0)}
    if gt is not None:out['gt']=gt.transpose(1,2,0)
    destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True);savemat(destination,out,do_compression=False)


if __name__=='__main__':main()
