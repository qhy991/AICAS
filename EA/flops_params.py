import numpy as np
def conv_comp(in_channel,out_channel,out_shape):
    flops = in_channel*3*3*out_channel*out_shape*out_shape
    params = in_channel*out_channel*3*3+out_channel
    return flops,params
def linear_comp(in_channel,out_channel):
    flops = in_channel*out_channel
    params = flops
    return flops,params
def pool_comp(channel,out_shape):
    flops = channel*out_shape*out_shape/4
    return flops
# todo 计算每一层每一次计算用到的memory
def flops_params(code):
    flops_list = []
    params_list = []
    channel_num = np.array(code[:5])*np.array([64,64,128,256,512])
    layer_num = code[-5:]
    map_pool = {0:True,1:True,2:False}
    pools = [map_pool[key] for key in code[10:16]] 
    shape_size = [32,32,16,8,4,2]
    # stage_0
    # for i in range(layer_num[0]):
    temp_flops,temp_params = conv_comp(3,channel_num[0],32)
    flops_list.append(temp_flops)
    params_list.append(temp_params)
    if int(layer_num[0]) == 2:
        temp_flops,temp_params = conv_comp(channel_num[0],channel_num[0],32)
        flops_list.append(temp_flops)
        params_list.append(temp_params)
    # stage1-4
    for i in range(1,5):
        if pools[i]:
            temp_flops = pool_comp(channel_num[i],shape_size[i])
            flops_list.append(temp_flops)
            for j in range(int(layer_num[i])):
                temp_flops,temp_params = conv_comp(channel_num[i],channel_num[i],shape_size[i])
                flops_list.append(temp_flops)
                params_list.append(temp_params)
        else:
            temp_flops,temp_params = conv_comp(channel_num[i-1],channel_num[i],shape_size[i])
            for j in range(int(layer_num[i]-1)):
                temp_flops,temp_params = conv_comp(channel_num[i],channel_num[i],shape_size[i])
                flops_list.append(temp_flops)
                params_list.append(temp_params)
    if pools[5]:
        temp_flops = pool_comp(channel_num[4],1)
        flops_list.append(temp_flops)
    temp_flops,temp_params = linear_comp(channel_num[4],10)
    flops_list.append(temp_flops)
    params_list.append(temp_params)   
    return sum(flops_list)/1e6,sum(params_list)/1e6

if __name__ == "__main__":
    import sys
    sys.path.append('../encode+decode/')
    from encode import encode
    import yaml
    from easydict import EasyDict
    config_path = "/home/qhy/Reserach/AICAS/config/cifar10/755-stage-2_1_6_7_3-ratio-0.75_0.75_0.5_0.125_1.0-op-repvgg_repvgg_repvgg_repvgg_repvgg-pool-False_True_True_True_True_True-pool_type-None_avgpool_maxpool_maxpool_maxpool_avgpool-cifar10.yaml"
    config = EasyDict(yaml.full_load(open(config_path)))
    code = encode(755,config)
    flops,params = flops_params(code)
    print(flops,params)