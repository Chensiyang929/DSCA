"""Sensor-response-aware domain head around an intact PrunedFusionNet."""
import torch
import torch.nn.functional as F
from torch import nn
from pruned_fusionnet import PrunedFusionNet
from head_aligners import resolve_domain


class ResponseAwareAligner(nn.Module):
    def __init__(self,bands):
        super().__init__();self.response_logits=nn.Parameter(torch.zeros(bands));self.band_gain=nn.Parameter(torch.ones(bands));self.dw=nn.Conv2d(bands,bands,3,padding=1,groups=bands);self.pw=nn.Conv2d(bands,bands,1);nn.init.zeros_(self.pw.weight);nn.init.zeros_(self.pw.bias)
    def forward(self,lms,pan):
        mean=lms.mean((2,3),keepdim=True);std=torch.sqrt(lms.var((2,3),keepdim=True,unbiased=False)+1e-6);z=(lms-mean)/std
        p=(pan-pan.mean((2,3),keepdim=True))/torch.sqrt(pan.var((2,3),keepdim=True,unbiased=False)+1e-6)
        proxy=(torch.softmax(self.response_logits,0)[None,:,None,None]*z).sum(1,keepdim=True)
        innovation=(p-proxy)*self.band_gain[None,:,None,None]
        return self.pw(F.gelu(self.dw(z+innovation)))


class ResponseGatedDomainWrappedPrunedFusionNet(nn.Module):
    def __init__(self,spectral_num=8,variant='head',initial_gate=.95):
        super().__init__();assert variant in ('base', 'shared', 'head');self.variant=variant;self.spectral_num=spectral_num;self.backbone=PrunedFusionNet(spectral_num)
        if variant == 'base':
            return
        count = 1 if variant == 'shared' else 2
        self.head=nn.ModuleList([ResponseAwareAligner(spectral_num) for _ in range(count)]);self.head_gate_raw=nn.Parameter(torch.full((count,spectral_num),float(initial_gate)))
    @staticmethod
    def _g(x,d):return x[d].clamp(0,1)[None,:,None,None]
    def forward(self,lms,pan,domain):
        d=resolve_domain(domain)
        if self.variant == 'base':
            return self.backbone(lms,pan)
        d = 0 if self.variant == 'shared' else d
        adjusted=lms-self._g(self.head_gate_raw,d)*self.head[d](lms,pan);return self.backbone(adjusted,pan)
    def backbone_parameters(self):return list(self.backbone.parameters())
    def adapter_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('backbone.')]


def load_pruned_initialization(model,state):model.backbone.load_state_dict(state)
