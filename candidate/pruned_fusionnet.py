"""Width-16 structured subnetwork of the successful width-32 FusionNet."""
from torch import nn


class Residual16(nn.Module):
 def __init__(self):super().__init__();self.conv20=nn.Conv2d(16,16,3,padding=1);self.conv21=nn.Conv2d(16,16,3,padding=1);self.relu=nn.ReLU(inplace=True)
 def forward(self,x):return x+self.conv21(self.relu(self.conv20(x)))


class PrunedFusionNet(nn.Module):
 def __init__(self,spectral_num=8):
  super().__init__();self.spectral_num=spectral_num;self.conv1=nn.Conv2d(spectral_num,16,3,padding=1);self.res1=Residual16();self.res2=Residual16();self.res3=Residual16();self.res4=Residual16();self.conv3=nn.Conv2d(16,spectral_num,3,padding=1);self.relu=nn.ReLU(inplace=True)
 def forward(self,x,pan,domain=None):
  z=self.relu(self.conv1(pan.repeat(1,self.spectral_num,1,1)-x))
  for block in [self.res1,self.res2,self.res3,self.res4]:z=block(z)
  return self.conv3(z)
 @property
 def receptive_field(self):return 21
