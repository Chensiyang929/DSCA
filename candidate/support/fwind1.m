function h = fwind1(hd,w1)
% Octave-compatible subset of MATLAB fwind1 for the uniform Huang-window case.
w1 = w1(:); n = length(w1);
t = (-(n-1)/2:(n-1)/2) * (2/(n-1));
[t1,t2] = meshgrid(t,t); t12 = sqrt(t1.*t1+t2.*t2);
w = interp1(t,w1,t12,'linear',0);
hd = rot90(fftshift(rot90(hd,2)),2);
h = fftshift(ifft2(hd));
if all(max(abs(imag(h))) < sqrt(eps)), h = real(h); end
h = rot90(h,2) .* w;
end
