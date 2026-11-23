"""Direct, learnable [0,1] identity gates around zero-init domain adapters."""
import torch
from torch import nn
from pruned_fusionnet import PrunedFusionNet
from head_aligners import CrossScaleDomainAligner, resolve_domain


class LinearGatedDomainWrappedPrunedFusionNet(nn.Module):
    def __init__(self, spectral_num=8, variant="head", initial_gate=0.95):
        super().__init__()
        assert variant in ('base', 'shared', 'head')
        self.variant=variant;self.spectral_num=spectral_num;self.backbone=PrunedFusionNet(spectral_num)
        if variant == 'base':
            return
        count = 1 if variant == 'shared' else 2
        self.head=nn.ModuleList([CrossScaleDomainAligner(spectral_num, domains=count) for _ in range(count)])
        self.head_gate_raw=nn.Parameter(torch.full((count,spectral_num),float(initial_gate)))

    @staticmethod
    def _gate(raw,domain):
        return raw[domain].clamp(0.0,1.0)[None,:,None,None]

    def forward(self,lms,pan,domain):
        domain=resolve_domain(domain)
        if self.variant == 'base':
            return self.backbone(lms,pan)
        domain = 0 if self.variant == 'shared' else domain
        residual=pan.repeat(1,self.spectral_num,1,1)-lms
        adjusted=lms-self._gate(self.head_gate_raw,domain)*self.head[domain](residual,domain)
        return self.backbone(adjusted,pan)

    def backbone_parameters(self):return list(self.backbone.parameters())
    def adapter_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('backbone.')]


def load_pruned_initialization(model,state):model.backbone.load_state_dict(state)
