% Score FR predictions and internal Wald diagnostics with a fixed sensor gain.
pkg load image; pkg load signal;
args=argv(); root=args{1}; sensor=args{2}; code=args{3}; alpha=str2double(args{4});
assert(isfinite(alpha));
toolbox=fullfile(code,'eval_toolbox');
addpath(fullfile(toolbox,'Tools')); addpath(fullfile(toolbox,'Quality_Indices'));
addpath(fullfile(code,'candidate','support')); cd(toolbox);
names={'PSNR','HQNR','Dlambda','Ds','Q2n','SAM','ERGAS','SCC','Qavg','SSIM'};
L=11; if strcmp(sensor,'GF2'),L=10;end
peak=2^L-1; scenes=0:19; if numel(args)>4, scenes=str2double(strsplit(args{5},',')); end
values=zeros(numel(scenes),10);
for row=1:numel(scenes)
 sc=scenes(row);
 folder=fullfile(root,sensor,'head',sprintf('scene%02d',sc));
 f=load(fullfile(folder,'full.mat')); r=load(fullfile(folder,'internal_wald.mat'));
 F=double(f.proposed); corrected=F+alpha*(double(f.I_MS)-MTF(F,sensor,4));
 [dl,ds,h]=indexes_evaluation_FS(corrected,double(f.I_MS_LR),double(f.I_PAN),L,0,double(f.I_MS),sensor,4,0);
 values(row,2:4)=[h,dl,ds];
 RR=double(r.proposed); gt=double(r.gt);
 pred=RR+alpha*(double(r.I_MS)-MTF(RR,sensor,4));
 [qavg,sam,ergas,scc,q2]=indexes_evaluation(pred,gt,4,L,32,1,21,0);
 pred=pred(22:end-21,22:end-21,:); gt=gt(22:end-21,22:end-21,:);
 psnr=10*log10(peak^2/max(mean((pred(:)-gt(:)).^2),1e-20));
 ss=zeros(1,size(pred,3));for b=1:size(pred,3),ss(b)=ssim(pred(:,:,b),gt(:,:,b));end
 values(row,[1,5,6,7,8,9,10])=[psnr,q2,sam,ergas,scc,qavg,mean(ss)];
 fprintf('%s scene%02d fixed_alpha=%.9f HQNR=%.8f\n',sensor,sc,alpha,h);fflush(stdout);
end
assert(all(isfinite(values(:))));
calibration_source='frozen_sensor_scalar_alpha_final'; official_rr=false;
save('-mat7-binary',fullfile(root,'metrics_mtf_fixed_scalar.mat'),'sensor','scenes','names','values','alpha','calibration_source','official_rr');
fid=fopen(fullfile(root,'summary_mtf_fixed_scalar.csv'),'w');fprintf(fid,'sensor,scenes,alpha');
for k=1:numel(names),fprintf(fid,',mean_%s,std_%s',names{k},names{k});end
fprintf(fid,'\n%s,%d,%.12g',sensor,numel(scenes),alpha);
for k=1:numel(names),fprintf(fid,',%.12g,%.12g',mean(values(:,k)),std(values(:,k)));end
fprintf(fid,'\n');fclose(fid);

