function D_s_index = D_s(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,q)
if nargin<6,S=32;end;if nargin<7,q=1;end
if any(size(I_F)~=size(I_MS)),error('The two images must have the same dimensions');end
[N,M,Nb]=size(I_F);if rem(N,S)~=0||rem(M,S)~=0,error('image dimensions must be multiples of block size');end
pan_filt=interp23tap(imresize(I_PAN,1./ratio),ratio);D_s_index=0;
for i=1:Nb
 band1=I_F(:,:,i);band2=I_PAN;fun=@(bs) robust_uqi(bs.data,band2(bs.location(1):bs.location(1)+S-1,bs.location(2):bs.location(2)+S-1));Q_high=mean2(blockproc(band1,[S S],fun));
 band1=I_MS(:,:,i);band2=pan_filt;fun=@(bs) robust_uqi(bs.data,band2(bs.location(1):bs.location(1)+S-1,bs.location(2):bs.location(2)+S-1));Q_low=mean2(blockproc(band1,[S S],fun));D_s_index=D_s_index+abs(Q_high-Q_low)^q;
end
D_s_index=(D_s_index/Nb)^(1/q);
end
function Q=robust_uqi(x,y)
x=double(x(:));y=double(y(:));mx=mean(x);my=mean(y);dx=x-mx;dy=y-my;n=max(numel(x)-1,1);cxx=sum(dx.*dx)/n;cyy=sum(dy.*dy)/n;cxy=sum(dx.*dy)/n;den=(cxx+cyy)*(mx^2+my^2);if den==0,Q=1;else,Q=4*cxy*mx*my/den;end
end
