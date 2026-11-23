import os,sys
from pathlib import Path
os.environ.setdefault('WALD_DEVICE','cuda:0');ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'support'))
import teacher_output_loader
from direct_teacher_loader import direct_teacher
from module_split_optimizers import install_module_optimizers


def main():
 teacher_output_loader.load_teacher_output=direct_teacher;source=(ROOT/'train_SDE_instrumented.py').read_text(encoding="utf-8");source=source.replace('from mymodel import FusionNet  # 学生模型','from linear_gated_domain_wrapped_pruned_fusionnet import LinearGatedDomainWrappedPrunedFusionNet as FusionNet, load_pruned_initialization');marker='model_student = FusionNet(bands).to(device)';replacement="model_student = FusionNet(bands, os.environ['WRAP_VARIANT'], float(os.environ.get('WRAP_INITIAL_GATE','0.95'))).to(device); load_pruned_initialization(model_student, torch.load(os.environ['PRUNED_PRETRAIN'],map_location=device,weights_only=True))";assert source.count(marker)==1;source=source.replace(marker,replacement)
 old='                        loss_spa = (loss_x + loss_y)/100';new="""                        with torch.no_grad():
                            lms_gra_x, lms_gra_y = sobel_filter(lms)
                            base_x, base_y = F_ms2pan(lms_gra_x, lms_gra_y)
                            base_spa = (loss_calculator.SDE_Loss(base_x, pan_gra_x) + loss_calculator.SDE_Loss(base_y, pan_gra_y)) / 2
                        loss_spa = torch.sqrt(((loss_x + loss_y) / 2) / (base_spa.detach() + 1e-8) + 1e-8)""";assert source.count(old)==1;source=source.replace(old,new);source=install_module_optimizers(source);exec(compile(source,'<linear_gated_domain_wrapped>','exec'),{'__name__':'__main__','__file__':str(ROOT/'train_SDE_instrumented.py')})


if __name__=='__main__':main()
