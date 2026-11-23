"""Width16 candidate with one universal dimensionless spatial-loss definition."""
import os
os.environ.setdefault('WALD_DEVICE','cuda:0')
from pathlib import Path
import sys
R=Path(__file__).resolve().parent;Z=R/'ZSVD-test-for-mix/zup';sys.path.insert(0,str(Z))
import teacher_output_loader
from direct_teacher_loader import direct_teacher
from split_optimizers import install_split_optimizers
def main():
 teacher_output_loader.load_teacher_output=direct_teacher
 source=(R/'train_SDE_instrumented.py').read_text();old='from mymodel import FusionNet  # 学生模型';assert source.count(old)==1;source=source.replace(old,'from pruned_fusionnet import PrunedFusionNet as FusionNet  # fixed width16 student')
 marker='model_student = FusionNet(bands).to(device)';assert source.count(marker)==1;source=source.replace(marker,marker+"; model_student.load_state_dict(torch.load(os.environ['PRUNED_PRETRAIN'],map_location=device,weights_only=True))")
 old_spatial='                        loss_spa = (loss_x + loss_y)/100'
 new_spatial="""                        with torch.no_grad():
                            lms_gra_x, lms_gra_y = sobel_filter(lms)
                            base_x, base_y = F_ms2pan(lms_gra_x, lms_gra_y)
                            base_spa = (loss_calculator.SDE_Loss(base_x, pan_gra_x) + loss_calculator.SDE_Loss(base_y, pan_gra_y)) / 2
                        loss_spa = torch.sqrt(((loss_x + loss_y) / 2) / (base_spa.detach() + 1e-8) + 1e-8)"""
 assert source.count(old_spatial)==1;source=source.replace(old_spatial,new_spatial)
 source=install_split_optimizers(source)
 exec(compile(source,'<pruned_normalized_spatial>','exec'),{'__name__':'__main__','__file__':str(R/'train_SDE_instrumented.py')})
if __name__=='__main__':main()
