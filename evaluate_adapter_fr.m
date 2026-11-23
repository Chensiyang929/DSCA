% Score an FR adapter variant with zero gain and with the fixed sensor gain.
pkg load image;
pkg load signal;
args=argv();
root=args{1}; sensor=args{2}; arm=args{3}; code=args{4};
alpha=str2double(args{5}); scenes=str2double(strsplit(args{6},','));
assert(isfinite(alpha) && all(isfinite(scenes)));
toolbox=fullfile(code,'eval_toolbox');
addpath(fullfile(toolbox,'Tools'));
addpath(fullfile(toolbox,'Quality_Indices'));
cd(toolbox);
L=11;
if strcmp(sensor,'GF2'), L=10; end
scores=zeros(numel(scenes),3,2);
for row=1:numel(scenes)
  f=load(fullfile(root,sensor,arm,sprintf('scene%02d',scenes(row)),'full.mat'));
  F=double(f.proposed); ms=double(f.I_MS); low_ms=double(f.I_MS_LR); pan=double(f.I_PAN);
  residual=ms-MTF(F,sensor,4);
  for mode=1:2
    prediction=F+(mode-1)*alpha*residual;
    [dl,ds,hqnr]=indexes_evaluation_FS(prediction,low_ms,pan,L,0,ms,sensor,4,0);
    scores(row,:,mode)=[hqnr,dl,ds];
  end
end
assert(all(isfinite(scores(:))));
names={'HQNR','Dlambda','Ds'};
modes={'off','fixed'};
save('-mat7-binary',fullfile(root,'fr_adapter_scores.mat'),'scores','sensor','arm','alpha','scenes','names','modes');
for mode=1:2
  fprintf('%s %s %s HQNR=%.8f Dlambda=%.8f Ds=%.8f\n',sensor,arm,modes{mode},mean(scores(:,:,mode),1));
end
