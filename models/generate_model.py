import numpy as np
from random import choices 

# the layer number of each stage
stage_layer_max = [2, 4, 14, 2]
# the channel number of each layer
layer_num_max = [64, 128, 256, 512]
# the descending step of channel 
step = 8 
op = ['repvgg','vgg'] # every stage
maxpool = [True,False]
ratio = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
datasets = ["cifar10","cifar100"]


# operation 
# dataset = choices(datasets)[0]
# class_num = 10 if dataset == "cifar10" else 100
import yaml
import os 
import collections
def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
 
def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
for n in range(100):
    config = {}
    # 随机生成模型
    stage_layer = [choices(range(1,stage_layer_max[0]+1))[0],choices(range(1,stage_layer_max[1]+1))[0],choices(range(1,stage_layer_max[2]+1))[0],choices(range(1,stage_layer_max[3]+1))[0]]
    # stage_layer
    stage_ratio = [choices(ratio)[0],choices(ratio)[0],choices(ratio)[0],choices(ratio)[0]]
    # stage_ratio
    with_maxpool = [choices(maxpool)[0],choices(maxpool)[0],choices(maxpool)[0],choices(maxpool)[0]]
    # with_maxpool
    op_type = [choices(op)[0],choices(op)[0],choices(op)[0],choices(op)[0],choices(op)[0]]
    dir = str(n)+'-stage-' + str(stage_layer[0]) + '_' + str(stage_layer[1]) + '_' + str(stage_layer[2]) + '_' + str(stage_layer[3]) + '-ratio-' \
        + str(stage_ratio[0]) + '_' + str(stage_ratio[1]) + '_' + str(stage_ratio[2]) + '_' + str(stage_ratio[3]) + \
            '-op-' + str(op_type[0]) + '_' + str(op_type[1]) + '_' + str(op_type[2]) + '_' + str(op_type[3]) + '_' + str(op_type[4]) + '-max-' + \
                str(with_maxpool[0]) + '_' + str(with_maxpool[1]) + '_' + str(with_maxpool[2]) + '_' + str(with_maxpool[3])

    config['model'] = {'stage_layer':stage_layer,'stage_ratio': stage_ratio ,'with_maxpool':with_maxpool,'op_type':op_type,"layer_num_max":layer_num_max}
    config['train'] = {'start_epoch': 0, "epochs": 300, "warmup_epochs": 20, "warmup_lr": 1.0e-2, 
                    "lr_scheduler":{"min_lr":1.0e-5,"decay_epochs":30,"decay_rate":0.1},"batch_size":64,"loss":"crossentropy","workers":4,
                    "optimizer":{"name":"SGD","base_lr":0.01,"repvgg_lr":0.001,"momentum":0.9,"eps":1.0e-8,"betas":[0.9,0.999],"weight_decay_param":{"base_decay":5e-4,"repvgg_decay":0.05,"echo":False}},
                    "label_smoothing":0.1,"clip_grad":0.0,"ema_alpha":0.0,"ema_update_period":8,"use_l2_norm":True,
                    "no_weight_decay": "rbr_dense"}
    config['val'] = {"batch_size":32,"workers":4} 
    config['output'] = {"print_freq":100,"epoch_print_freq":1,"save_freq":20,
                        "dir": os.path.join("./log",dir), "name": str(n)}
    save_path = os.path.join("./config",dir+'.yaml')
    save_dict_to_yaml(config,save_path)


