import os
import sys
sys.path.append("..")
from thop import profile 
from models import model as M
from utils import repvgg_model_convert
import yaml
import torch
def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True,default_flow_style=False))
def decode(vars,dataset,n=0,save=True,constrain="acc"):
    mapping_op = {0:'repvgg',1:'vgg'}
    mapping_pool = {0:'maxpool',1:'avgpool',2:"None"}
    mapping_with_pool = {0:True,1:True,2:False}
    stage_ratio = [vars[i]/8 for i in range(5)]
    for ratio in stage_ratio:
        print("- ",ratio)
    stage_layer = [vars[i] for i in range(-5,0)]
    for layer in stage_layer:
        print("- ",layer)
    pool = [None]+[mapping_pool[vars[i]] for i in range(11,16) ]
    with_pool = [False]+[mapping_with_pool[vars[i]] for i in range(11,16) ]
    op_type = [mapping_op[vars[i]] for i in range(5,10) ]
    layer_num_max = [64, 64, 128, 256, 512]
    dir = str(n)+'-stage-' + str(stage_layer[0]) + '_' + str(stage_layer[1]) + '_' + str(stage_layer[2]) + '_' + str(stage_layer[3]) + '_' + str(stage_layer[4]) + '-ratio-' \
    + str(stage_ratio[0]) + '_' + str(stage_ratio[1]) + '_' + str(stage_ratio[2]) + '_' + str(stage_ratio[3]) + '_' + str(stage_ratio[4]) +  \
        '-op-' + str(op_type[0]) + '_' + str(op_type[1]) + '_' + str(op_type[2]) + '_' + str(op_type[3]) + '_' + str(op_type[4]) + '-pool-' + \
            str(with_pool[0]) + '_' + str(with_pool[1]) + '_' + str(with_pool[2]) + '_' + str(with_pool[3]) + '_' + str(with_pool[4]) + '_' + str(with_pool[5]) + '-pool_type-'+\
                str(pool[0]) + '_' + str(pool[1]) + '_' + str(pool[2]) + '_' + str(pool[3]) + '_' + str(pool[4]) + '_' + str(pool[5]) + '-' + dataset
    config = {}
    config['dataset'] = dataset
    config['model'] = {'stage_layer':stage_layer,'stage_ratio': stage_ratio ,'with_pool':with_pool,"pool_type":pool,'op_type':op_type,"layer_num_max":layer_num_max}
    config['train'] = {'start_epoch': 0, "epochs": 100, "warmup_epochs": 0, "warmup_lr": 1.0e-2, 
                    "lr_scheduler":{"min_lr":1.0e-6,"decay_epochs":30,"decay_rate":0.1},"batch_size":256,"loss":"crossentropy","workers":4,
                    "optimizer":{"name":"SGD","base_lr":0.35,"repvgg_lr":0.3,"momentum":0.9,"eps":1.0e-8,"betas":[0.9,0.999],"weight_decay_param":{"base_decay":5e-4,"repvgg_decay":1e-4,"echo":False}},
                    "label_smoothing":0.1,"clip_grad":0.0,"ema_alpha":0.0,"ema_update_period":8,"use_l2_norm":True,
                    "no_weight_decay": ["rbr_dense","rbr_1x1"]}
    config['val'] = {"batch_size":128,"workers":4} 
    config['output'] = {"print_freq":100,"epoch_print_freq":1,"save_freq":20,
                    "dir": os.path.join("./log/search-best",dir), "name": str(n)}
    save_path = os.path.join("../config/search-best",dir+'.yaml')
    
    model = M.Net(config, 10)
    # print(model)
    model = repvgg_model_convert(model)
    input = torch.randn(1, 3, 32, 32)
    flops,params = profile(model, inputs=(input, ))
    md_fmt = "|"+ str(n)+'|' + str(stage_layer[0]) + '-' + str(stage_layer[1]) + '-' + str(stage_layer[2]) + '-' + str(stage_layer[3]) + '-' + str(stage_layer[4]) + '|'\
        + str(stage_ratio[0]) + '-' + str(stage_ratio[1]) + '-' + str(stage_ratio[2]) + '-' + str(stage_ratio[3]) + '-' + str(stage_ratio[4]) + \
            '|' + str(op_type[0]) + '-' + str(op_type[1]) + '-' + str(op_type[2]) + '-' + str(op_type[3]) + '-' + str(op_type[4]) + '|' + \
                str(with_pool[0]) + '-' + str(with_pool[1]) + '-' + str(with_pool[2]) + '-' + str(with_pool[3]) + '-' + str(with_pool[4]) + '|' +\
                str(pool[0]) + '-' + str(pool[1]) + '-' + str(pool[2]) + '-' + str(pool[3]) + '-' + str(pool[4]) + '-' + str(pool[5]) + '|' + dataset+ "|" + str(round(flops/1e6,2))+"|"+str(round(params/1e6,2))+"|"+"|"+constrain+"|"
    if save:
        save_dict_to_yaml(config,save_path)
    print(md_fmt)

def decode_lite(vars,dataset):
    mapping_op = {0:'repvgg',1:'vgg'}
    mapping_pool = {0:'maxpool',1:'avgpool',2:"None"}
    mapping_with_pool = {0:True,1:True,2:False}
    stage_ratio = [vars[i] for i in range(5)]
    stage_layer = [vars[i] for i in range(-5,0)]
    pool = [None]+[mapping_pool[vars[i]] for i in range(10,16) ]
    with_pool = [False]+[mapping_with_pool[vars[i]] for i in range(10,16) ]
    op_type = [mapping_op[vars[i]] for i in range(5,10) ]
    layer_num_max = [64, 64, 128, 256, 512]
    config = {}
    config['dataset'] = dataset
    config['model'] = {'stage_layer':stage_layer,'stage_ratio': stage_ratio ,'with_pool':with_pool,"pool_type":pool,'op_type':op_type,"layer_num_max":layer_num_max}
    if dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100
    # model = M.Net(config, num_classes)
    # # print(model)
    # model = repvgg_model_convert(model)
    # input = torch.randn(1, 3, 32, 32)
    # flops,params = profile(model, inputs=(input, ))
    # dir = 'stage-' + str(stage_layer[0]) + '_' + str(stage_layer[1]) + '_' + str(stage_layer[2]) + '_' + str(stage_layer[3]) + '_' + str(stage_layer[4]) + '-ratio-' \
    # + str(stage_ratio[0]) + '_' + str(stage_ratio[1]) + '_' + str(stage_ratio[2]) + '_' + str(stage_ratio[3]) + '_' + str(stage_ratio[4]) +  \
    #     '-op-' + str(op_type[0]) + '_' + str(op_type[1]) + '_' + str(op_type[2]) + '_' + str(op_type[3]) + '_' + str(op_type[4]) + '-pool-' + \
    #         str(with_pool[0]) + '_' + str(with_pool[1]) + '_' + str(with_pool[2]) + '_' + str(with_pool[3]) + '_' + str(with_pool[4]) + '_' + str(with_pool[5]) + '-pool_type-'+\
    #             str(pool[0]) + '_' + str(pool[1]) + '_' + str(pool[2]) + '_' + str(pool[3]) + '_' + str(pool[4]) + '_' + str(pool[5]) 
    # return dir,flops/1e6,params/1e6
    return config
    
# decode(list(res['Vars'][0]),"cifar10",1)  
        