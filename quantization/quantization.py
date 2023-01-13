import torchvision.models as models  # for example model
from mqbench.prepare_by_platform import prepare_by_platform  # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType  # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration  # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization  # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy  # remove quant nodes for deploy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import sys

sys.path.append("..")
from models import model as M
from easydict import EasyDict
import yaml
import numpy as np
import os
import tqdm
from data import build_loader, build_hs_loader
from logger import create_logger
from lr_scheduler import build_lr_scheduler
from models import model as M
from optimizer import build_optimizer
from utils import seed_all, save_checkpoint, train_one_epoch, validate, get_grad_norm, get_dataset,repvgg_model_convert
# from torch.utils.tensorboard import SummaryWriter

config_path = "/home/qhy/Reserach/AICAS/config/cifar10/869-stage-2_1_4_8_1-ratio-0.25_0.25_0.25_0.5_0.625-op-vgg_repvgg_repvgg_vgg_repvgg-pool-False_False_True_False_False_False-pool_type-None_None_avgpool_None_None_None-cifar10.yaml"
config = EasyDict(yaml.full_load(open(config_path)))

train_loader, val_loader, num_classes = get_dataset(config)

logger = None
writer = None
model_list = os.listdir("../config/cifar10/")
acc_f32 = {}
acc_int8 = {}
for model_path in tqdm.tqdm(model_list):
    # print(model_path)
    config_path = os.path.join("../config/cifar10/",model_path)
    
    if model_path not in ["repvgg.yaml","vgg.yaml","vgg16-stage-2_2_3_3_3-ratio-1.0_1.0_1.0_1.0_1.0-op-vgg_vgg_vgg_vgg_vgg-max-False_True_True_True_True.yaml",
                          "stage-1_3_4_1-ratio-1.0_0.375_0.75_0.125-op-vgg_vgg_repvgg_vgg-max-True_False_True_False.yaml",
                          "repvgg-A0-stage-2_2_3_3_3-ratio-1.0_1.0_1.0_1.0_1.0-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False.yaml",
                          "benchmark-vgg11-stage-1_1_2_2_2-ratio-1.0_1.0_1.0_1.0_1.0-op-vgg_vgg_vgg_vgg_vgg-max-False_True_True_True_True-cifar10.yaml",
                          "benchmark-vgg13-stage-2_2_2_2_2-ratio-1.0_1.0_1.0_1.0_1.0-op-vgg_vgg_vgg_vgg_vgg-max-False_True_True_True_True-cifar10.yaml",
                          "benchmark-vgg16-stage-2_2_3_3_3-ratio-1.0_1.0_1.0_1.0_1.0-op-vgg_vgg_vgg_vgg_vgg-max-False_True_True_True_True-cifar10.yaml",
                          "benchmark-vgg19-stage-2_2_4_4_4-ratio-1.0_1.0_1.0_1.0_1.0-op-vgg_vgg_vgg_vgg_vgg-max-False_True_True_True_True-cifar10.yaml",
                          "benchmark-repvgg-a0-stage-1_2_4_14_1-ratio-0.75_0.75_0.75_0.75_2.5-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-a1-stage-1_2_4_14_1-ratio-1.0_1.0_1.0_1.0_2.5-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-a2-stage-1_2_4_14_1-ratio-1.5_1.5_1.5_1.5_2.75-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-b0-stage-1_2_4_14_1-ratio-1.0_1.0_1.0_1.0_2.75-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-b1-stage-1_2_4_14_1-ratio-2.0_2.0_2.0_2.0_4.0-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-b2-stage-1_2_4_14_1-ratio-2.5_2.5_2.5_2.5_5.0-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-b3-stage-1_2_4_14_1-ratio-3.0_3.0_3.0_3.0_5.0-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10.yaml",
                          "benchmark-repvgg-a0-stage-1_2_4_14_1-ratio-0.75_0.75_0.75_0.75_2.5-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10-lr.yaml",
                          "benchmark-repvgg-a1-stage-1_2_4_14_1-ratio-1.0_1.0_1.0_1.0_2.5-op-repvgg_repvgg_repvgg_repvgg_repvgg-max-False_False_False_False_False-cifar10-lr.yaml"]:
        
        model_path_2 = "/home/qhy/Reserach/AICAS/log/cifar10/" + config_path.split(
            '/')[-1].replace(".yaml", "/best_ckpt.pth")
        num = int(model_path.split("-")[0])

        if os.path.exists(model_path_2) and (742<=num):
            print(num)
            config = EasyDict(yaml.full_load(open(config_path)))
            model = M.Net(config, num_classes)
            model.load_state_dict(torch.load(model_path_2)['model'])
            model = repvgg_model_convert(model)
            model.eval()
            model.cuda()
            acc1, acc5, loss = validate(config, val_loader, model, logger, num_classes, 1,writer)    
            acc_f32[num] = acc1              
            print("=" * 100)
            print(acc1, acc5, loss)
            print("=" * 100)
            extra_config = {
                'extra_qconfig_dict': {
                    'w_observer': 'MSEObserver',                              # custom weight observer
                    'a_observer': 'MSEObserver',                              # custom activation observer
                    'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
                    'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
                    'w_qscheme': {
                        'bit': 8,                                             # custom bitwidth for weight,
                        'symmetry': False,                                    # custom whether quant is symmetric for weight,
                        'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for weight,
                        'pot_scale': True,                                   # custom whether scale is power of two for weight.
                    },
                    'a_qscheme': {
                        'bit': 8,                                             # custom bitwidth for activation,
                        'symmetry': False,                                    # custom whether quant is symmetric for activation,
                        'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for activation,
                        'pot_scale': True,                                   # custom whether scale is power of two for activation.
                    }
                }
            }
            backend = BackendType.Vitis
            model = prepare_by_platform(model, backend)
            # calibration loop
            enable_calibration(model)
            model.cuda()
            for idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.cuda(), labels.cuda()
                outputs = model(imgs)
            # torch.save(model.state_dict(),"./899-calibration.pth")
            enable_quantization(model)

            for idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.cuda(), labels.cuda()
                outputs = model(imgs)
            input_shape = {'data': [1, 3, 32, 32]}
            os.makedirs('./qmodel-pot/'+str(num),exist_ok = True)
            torch.save(model.state_dict(),"./qmodel-pot/"+str(num)+'/'+str(num)+'-qot.pth')
            convert_deploy(model, backend, input_shape,output_path='./qmodel-pot/'+str(num),model_name=str(num)+ "-q-vitis-best")
            torch.save(model.state_dict(),'./qmodel/'+str(num)+ "-q-vitis-deploy.pth")
            acc1, acc5, loss = validate(config, val_loader, model, logger, num_classes, 1,writer)
            acc_int8[num] = acc1
print(acc_int8,acc_f32)
np.save("acc_32_742-989-pot.npy",acc_f32)
np.save("acc_8_742-989-pot.npy",acc_int8)
