"""Scale-specific input alignment modules for DSCA."""

import torch
import torch.nn.functional as F
from torch import nn


def resolve_domain(domain):
    """Map an explicit semantic domain name to the corresponding Head index."""
    if domain == "gt":
        return 0
    if domain == "full":
        return 1
    raise ValueError("domain must be explicitly set to 'gt' or 'full'")


class CrossScaleDomainAligner(nn.Module):
    def __init__(self, bands, domains=2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(domains, bands))
        self.beta = nn.Parameter(torch.zeros(domains, bands))
        self.dw = nn.Conv2d(bands, bands, 3, padding=1, groups=bands)
        self.pw = nn.Conv2d(bands, bands, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, residual, domain):
        mean = residual.mean((2, 3), keepdim=True)
        std = torch.sqrt(residual.var((2, 3), keepdim=True, unbiased=False) + 1e-6)
        normalized = (residual - mean) / std
        normalized = (
            normalized * self.gamma[domain][None, :, None, None]
            + self.beta[domain][None, :, None, None]
        )
        return self.pw(F.gelu(self.dw(normalized)))
