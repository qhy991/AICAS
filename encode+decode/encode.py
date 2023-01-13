import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import os
from easydict import EasyDict
import yaml
# acc = np.zeros(740)
data = {}

def encode(num,config):
    mapping_op = {'repvgg':0,'vgg':1}
    mapping_pool = {'maxpool':0,'avgpool':1,"None":2}
    code = []
    if config['model']["stage_ratio"] != None:
        code += config['model']["stage_ratio"]
    else:
        print("Error in stage ration")
    
    if config['model']['op_type'] != None:
        code += [mapping_op[i] for i in config['model']['op_type']]
    else:
        print("Error in op type")
    
    if 509<num<642:
        mapping_pool = {True:0,False:2}
        if config['model']['with_pool'] != None:
            code += [mapping_pool[i] for i in config['model']['with_pool']]
            code += [0]
            print("maxpool",code)
        else:
            print("Error in maxpool")
    elif 641<num<650:
        mapping_pool = {True:1,False:2}
        if config['model']['with_pool'] != None:
            code += [mapping_pool[i] for i in config['model']['with_pool']]
            code += [1]
        else:
            print("Error in maxpool")
    elif 649<num:
        mapping_pool = {"maxpool":0,"avgpool":1,None:2}
        if config['model']['pool_type'] != None:
            code += [mapping_pool[i] for i in config['model']['pool_type']]
        else:
            print("Error in maxpool")
    if config['model']['stage_layer'] != None:
        code += config['model']['stage_layer']
    else:
        print("Error in stage layer")
    assert len(code) == 21,print(num,len(code),code)
    return code
if __name__ == "__main__":
    model_list = os.listdir("./config/cifar10")
    for model_path in tqdm.tqdm(model_list):
        # print(model_path)
        config_path = os.path.join("./config/cifar10/",model_path)
        model_path_2 = os.path.join("./log/cifar10/",model_path.replace(".yaml",""),"best_ckpt.pth")
        
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
            num = int(model_path.split("-")[0])
            # print(num)
            if os.path.exists(model_path_2) and (509<num):
                param = torch.load(model_path_2)
                acc = param['max_accuracy']
                if(acc>0.1):
                    config = EasyDict(yaml.full_load(open(config_path)))
                    code = encode(config=config,num=num)
                    data[num] = {"code":code,"acc":acc}
                    print(num,"done")
    # code_dataset = []
    # acc_dataset = []
    # for k in data.keys():
    #     code_dataset.append(data[k]['code'])
    #     acc_dataset.append(data[k]['acc'])
    np.save("/home/qhy/Reserach/AICAS/predictor/dataset/data-dict-980.npy",data)
    # np.save("/home/qhy/Reserach/AICAS/predictor/dataset/acc-dict-980.npy",np.array(acc_dataset))               
                    
                
