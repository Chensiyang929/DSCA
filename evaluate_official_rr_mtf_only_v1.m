% Official RR evaluation of the complete method: fixed-scalar MTF final output.
pkg load image; pkg load signal;
args=argv(); root=args{1}; pairs=args{2}; sensor=args{3}; code=args{4}; alpha=str2double(args{5});
addpath(fullfile(code,'eval_toolbox','Tools'));
addpath(fullfile(code,'eval_toolbox','Quality_Indices')); cd(fullfile(code,'eval_toolbox'));
names={'PSNR','Q2n','Qindex','SAM','ERGAS','SCC','SSIM'}; scenes=0:19; if numel(args)>5, scenes=str2double(strsplit(args{6},',')); end
L=11;if strcmp(sensor,'GF2'),L=10;end; peak=2^L-1;
values=zeros(numel(scenes),7);
for row=1:numel(scenes)
 sc=scenes(row);
 p=load(fullfile(pairs,sprintf('scene%02d.mat',sc)));
 r=load(fullfile(root,sprintf('scene%02d',sc),'adapted_rr.mat'));
 F=double(p.proposed); gt=double(p.gt);
 pred=F+alpha*(double(r.I_MS)-MTF(F,sensor,4));
 [q,sam,ergas,scc,q2]=indexes_evaluation(pred,gt,4,L,32,1,21,0);
 x=pred(22:end-21,22:end-21,:); y=gt(22:end-21,22:end-21,:);
 psnr=10*log10(peak^2/max(mean((x(:)-y(:)).^2),1e-20)); ss=zeros(1,size(x,3));
 for b=1:size(x,3),ss(b)=ssim(x(:,:,b),y(:,:,b));end
 values(row,:)=[psnr,q2,q,sam,ergas,scc,mean(ss)];
end
assert(all(isfinite(values(:))));
correction='fixed_scalar'; official_rr=true;
save('-mat7-binary',fullfile(root,'official_rr_metrics_fixed_scalar.mat'),'sensor','scenes','names','values','correction','alpha','official_rr');
fprintf('OFFICIAL_RR_MTF_ONLY %s PSNR=%.6f alpha=%.9f\n',sensor,mean(values(:,1)),alpha);

