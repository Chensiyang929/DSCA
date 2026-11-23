function out = blockproc(A, block_size, fun)
% Minimal MATLAB-compatible non-overlapping BLOCKPROC for DLPan metrics.
bh=block_size(1);bw=block_size(2);[h,w]=size(A);nr=floor(h/bh);nc=floor(w/bw);cells=cell(nr,nc);
for r=1:nr
 for c=1:nc
  y=(r-1)*bh+1;x=(c-1)*bw+1;bs=struct('data',A(y:y+bh-1,x:x+bw-1),'location',[y x]);cells{r,c}=fun(bs);
 end
end
sample=cells{1,1};if isscalar(sample),out=zeros(nr,nc);for r=1:nr,for c=1:nc,out(r,c)=cells{r,c};end,end;else,out=cell2mat(cells);end
end
