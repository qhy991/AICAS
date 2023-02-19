import numpy as np
def model_analysis(code, pe, simd, r_shift, classify):
    config_dic = {}
    
    channel_num = np.array(code[:5])*np.array([64, 64, 128, 256, 512])
    #print(channel_num)
    layer_num = code[-5:]
    #print(layer_num)
    map_pool = {0:True, 1:True, 2:False}
    pools = [map_pool[key] for key in code[10: 16]]
    pools_type = code[10: 16] 
    #print(pools)
    shape_size = [32, 32, 16, 8, 4, 2]
    
    # initial channel
    in_channel = 3
    out_channel = int(channel_num[0])
    
    param_num = 0
    
    for i in range(5):
        for j in range(int(layer_num[i])):
            name = "conv_"+str(i)+"_"+str(j)
            # initial config
            config = {'k':3, 's':1, 'p':1, 'in_shape':(3, 32, 32), 'out_shape':(8, 32, 32)}
            
            # modify 'in_shape'
            config['in_shape'] = (in_channel, shape_size[i], shape_size[i])
            
            if (j == int(layer_num[i])-1):
                if (not pools[i]) :
                    # no pooling 
                    config['s'] = 2
                    config['out_shape'] = (out_channel, shape_size[i+1], shape_size[i+1])
                    config['simd'] = simd[param_num]
                    config['pe'] = pe[param_num]
                    config['r_shift'] = r_shift[param_num]
                    config_dic[name]=config

                else :
                    # pooling
                    config['s'] = 1
                    config['out_shape'] = (out_channel, shape_size[i], shape_size[i])
                    config['simd'] = simd[param_num]
                    config['pe'] = pe[param_num]
                    config['r_shift'] = r_shift[param_num]
                    config_dic[name]=config
                    
                    if(pools_type[i]==0):
                        pool_name = "maxpool_"+str(i)
                        config_dic[pool_name]={'k':2, 's':1, 'p':0}
                    elif (pools_type[i]==1):
                        pool_name = "avgpool_"+str(i)
                        config_dic[pool_name]={'k':2, 's':1, 'p':0}

                # renew channel
                if i < 4 :
                    in_channel = out_channel
                    out_channel = int(channel_num[i+1])
                else :
                    in_channel = out_channel
                    out_channel = out_channel
                
                param_num += 1
                
            else :
                config['s'] = 1
                config['out_shape'] = (out_channel, shape_size[i], shape_size[i])
                config['simd'] = simd[param_num]
                config['pe'] = pe[param_num]
                config['r_shift'] = r_shift[param_num]
                config_dic[name]=config

                # renew channel
                in_channel = out_channel
                out_channel = int(channel_num[i])
                
                param_num += 1
    

            
    return config_dic
class ParamProcess:
    def __init__(self, name, config, w, bias, w_bit, in_bit, out_bit, bias_bit, pe, simd, r_shift, conv_linear=True):
        self.name = name
        self.config = config
        self.w = w
        self.bias = bias
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.bias_bit = bias_bit
        self.pe = pe
        self.simd = simd
        self.r_shift = r_shift
        self.conv_linear = conv_linear
    
    # 将矩阵整理成所需的存储样式
    # 转化为pe * tiles矩阵
    def w_to_hls_array(self, w):
        # print(#"w shape: ", w.shape)
        assert w.shape[0] % self.pe == 0, 'out_ch mod pe must 0' + "out_ch ="+str(w.shape[0]) +"pe = "+str(self.pe)
        # w 矩阵宽 k*K*in_ch
        h = w.shape[1]
        # res0 size = out_ch, k*K*in_ch // simd + (0 or 1)
        res0 = [[0 for i in range(h // self.simd)] for j in range(w.shape[0])]
        for out_ch in range(w.shape[0]):
            for i in range(h // self.simd):
                arr = w[out_ch][i*self.simd:(i+1)*self.simd]
                res0[out_ch][i] = array_to_string(arr, self.w_bit)
        
        # 处理不够整除部分
        if h % self.simd != 0:
            print('h mod simd != 0')
            print("h =",h,"simd =",self.simd)
            for out_ch in range(w.shape[0]):
                arr = w[out_ch][h // self.simd * self.simd]
                res0[out_ch].append(array_to_string(arr, self.w_bit))
                
        
        tiles = len(res0[0]) * (len(res0) // self.pe)
        self.w_tiles = tiles
        # print('tiles', tiles)
        res = [[0 for i in range(tiles)] for i in range(self.pe)]
        
        tiles_cnt = 0
        for i in range(len(res0) // self.pe):
            for j in range(len(res0[0])):
                for pe_cnt in range(self.pe):
                    res[pe_cnt][tiles_cnt] = res0[i * self.pe + pe_cnt][j]
                tiles_cnt += 1
        return res
    
    def bias_to_hls_array(self, bias):
        bias = bias.reshape(-1, self.pe)
        bias = bias.T
        
        return bias
    
    def conv(self):
        w = self.w
        bias = self.bias
        # w是二维矩阵形式
        conv_w = w.transpose(0, 2, 3, 1)
        # 处理为二维矩阵
        conv_w = conv_w.reshape(conv_w.shape[0], -1)
        # print(w.shape)
        # 先把w处理为每个元素位宽都是simd * w_bit形式
        conv_w = self.w_to_hls_array(conv_w)
        
        bias = self.bias_to_hls_array(bias)
        
        self.hls_w = conv_w
        self.hls_bias = bias
        
        return conv_w, bias
    
    def linear(self):
        w = self.w
        bias = self.bias
        # 若上一层是卷积层，需要调整参数位置
        if (self.conv_linear == True):
            last_conv_shape = self.config["last_layer_shape"]
            w = w.reshape(w.shape[0], last_conv_shape[0], last_conv_shape[1], last_conv_shape[2])
            w = w.transpose(0, 2, 3, 1)
            w = w.reshape(w.shape[0], -1)
        w = self.w_to_hls_array(w)
        bias = self.bias_to_hls_array(bias)

        self.hls_w = w
        self.hls_bias = bias

        return w, bias
    
    def w_to_hls_init_str(self, w) -> str:
        w_mem_type = "const ap_uint<"+str(self.w_bit * self.simd)+">"
        
        res = '//'  + self.name + '_w\n'
        res += '//PEs = %d, SIMD = %d\n' % (self.pe, self.simd)
        res += '//bit = %d\n' % self.w_bit
        res += w_mem_type
        res += (' ' + self.name + '_w')
        res += '[%d][%d] = {\n' % (len(w), len(w[0]))
        
        res += ",\n".join(map(lambda pe:"{\""+("\", \"".join(map(hex, pe)))+"\"}", w))
        res += '};\n'
        
        return res
    
    def bias_to_hls_init_str(self, bias) -> str:
        bias_bit_width = self.bias_bit
        
        w_mem_type = "const ap_int<"+str(self.bias_bit)+">"
        
        res = '// bias\n'
        res += '//'  + self.name + '_bias\n'
        res += '//w_bit = %d\n' % bias_bit_width
        res += w_mem_type
        res += (' ' + self.name + '_bias')
        res += '[%d][%d] = {\n' % (len(bias), len(bias[0]))
        
        res += ",\n".join(map(lambda pe:"{\""+("\", \"".join(map(hex, pe)))+"\"}", bias))
        res += '};\n'
        
        return res
    
    def layer_param_to_init_str(self, w, bias) -> str:
        res = self.w_to_hls_init_str(w)
        res += self.bias_to_hls_init_str(bias)
        
        return res
    
    def add_a_config_str(self, config_name, value) -> str:
        res = '#define %s_%s %d \n' % (self.name.upper(), config_name.upper(), value)
        return res

    def conv_config_str(self) -> str:
        res = '// ' + self.name + '\n'
        res += self.add_a_config_str('K', self.config['k'])
        res += self.add_a_config_str('S', self.config['s'])
        res += self.add_a_config_str('P', self.config['p'])
        res += self.add_a_config_str('IFM_CH', self.config['in_shape'][0])
        res += self.add_a_config_str('IFM_ROW', self.config['in_shape'][1])
        res += self.add_a_config_str('IFM_COL', self.config['in_shape'][2])

        res += self.add_a_config_str('OFM_CH', self.config['out_shape'][0])
        res += self.add_a_config_str('OFM_ROW', self.config['out_shape'][1])
        res += self.add_a_config_str('OFM_COL', self.config['out_shape'][2])

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)
        res += self.add_a_config_str('BIAS_BIT', self.bias_bit)

        res += self.add_a_config_str('R_SHIFT', self.r_shift)

        res += '\n'

        return res
    
    def linear_config_str(self) -> str:
        # conv1x1
        res = '// ' + self.name + '\n'
        res += self.add_a_config_str('K', 1)
        res += self.add_a_config_str('S', 1)
        res += self.add_a_config_str('P', 0)
        res += self.add_a_config_str('IFM_CH', self.config['in_shape'])
        res += self.add_a_config_str('IFM_ROW', 1)
        res += self.add_a_config_str('IFM_COL', 1)

        res += self.add_a_config_str('OFM_CH', self.config['out_shape'])
        res += self.add_a_config_str('OFM_ROW', 1)
        res += self.add_a_config_str('OFM_COL', 1)

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)
        res += self.add_a_config_str('BIAS_BIT', self.bias_bit)

        res += self.add_a_config_str('R_SHIFT', self.r_shift)

        res += '\n'

        return res

def config_param(code, config_dic):
    channel_num = np.array(code[:5])*np.array([64, 64, 128, 256, 512])
    layer_num = code[-5:]
    
    conv = ""
    conv_config = ""
    
    # conv layer
    for i in range(5):
        for j in range(int(layer_num[i])):
            name = "conv_"+str(i)+"_"+str(j)
            config = config_dic[name]
            
            weight_shape = (int(config['out_shape'][0]), int(config['in_shape'][0]), 3, 3)
            weight = np.random.randint(-256, 255, size=weight_shape)
            #print(weight.shape)
            
            bias_shape = (int(config['out_shape'][0]))
            bias = np.random.randint(-256, 255, size=bias_shape)
            
            processer = ParamProcess(
                name=name, 
                config=config,
                w=weight, bias=bias, 
                w_bit=8, in_bit=8, out_bit=8, bias_bit=32, 
                pe=config['pe'], simd=config['simd'], r_shift=config['r_shift'])
            
            w_str, b_str = processer.conv()
            conv_str = processer.layer_param_to_init_str(w_str, b_str)
            conv_config_str = processer.conv_config_str()
            
            conv += conv_str
            conv_config += conv_config_str
    

    
    return conv, conv_config
def bram_inference(code, config_dic):
    channel_num = np.array(code[:5])*np.array([64, 64, 128, 256, 512])
    layer_num = code[-5:]
    shape_size = [32,32,16,8,4,2]
    
    # initial
    bram_num = np.ceil(64/18)

    
    
    first_conv = 0

    for name in config_dic.keys():
        if "conv" in name :
            # first layer
            if (first_conv == 0):
                bram_num += np.ceil(64*config_dic[name]['in_shape'][0]/18)+np.ceil(8*config_dic[name]['in_shape'][0]/18)
                first_conv += 1
                
            # other
            # weight mem
            bram_num += config_dic[name]['pe']* np.ceil(config_dic[name]['simd']*8/18)*np.ceil(((config_dic[name]['in_shape'][0]*9)/config_dic[name]['simd'])*(config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # bias mem
            bram_num += config_dic[name]['pe']* np.ceil(8/18)*np.ceil((config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # padding fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # swu fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # conv_out fifo
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)

        elif "maxpool" in name :
            # maxpool
            # swu_out
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            # pool_out
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            
        elif "avgpool" in name :
            # avgpool
            # swu_out
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            # pool_out
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)

        elif "linear" in name :
            # linear(conv1x1)
            # weight mem
            bram_num += config_dic[name]['pe']* np.ceil(config_dic[name]['simd']*8/18)*np.ceil(((config_dic[name]['in_shape'][0]*9)/config_dic[name]['simd'])*(config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # bias mem
            bram_num += config_dic[name]['pe']* np.ceil(8/18)*np.ceil((config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # padding fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # swu fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # conv_out fifo
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            
    bram_num += np.ceil(64/18)
    
    return bram_num

def hw_inference(code, config_dic):
    channel_num = np.array(code[:5])*np.array([64, 64, 128, 256, 512])
    layer_num = code[-5:]
    shape_size = [32,32,16,8,4,2]
    
    # initial
    bram_num = np.ceil(64/18)

    dsp_conv = 0
    dsp_all = 0

    cycle_conv = 0
    cycle_all = 0
    
    first_conv = 0
    pre_name = ""
    for name in config_dic.keys():
        if "conv" in name :
            # first layer
            if (first_conv == 0):
                bram_num += np.ceil(64*config_dic[name]['in_shape'][0]/18)+np.ceil(8*config_dic[name]['in_shape'][0]/18)
                first_conv += 1
                
            # other
            # weight mem
            bram_num += config_dic[name]['pe']* np.ceil(config_dic[name]['simd']*8/18)*np.ceil(((config_dic[name]['in_shape'][0]*9)/config_dic[name]['simd'])*(config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # bias mem
            bram_num += config_dic[name]['pe']* np.ceil(8/18)*np.ceil((config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # padding fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # swu fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # conv_out fifo
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            pre_name = name

            if config_dic[name]['simd'] == 3:
                dsp_conv = config_dic[name]['pe']*2
            elif config_dic[name]['simd'] == 4:
                dsp_conv = config_dic[name]['pe']*2
            elif config_dic[name]['simd'] == 8:
                dsp_conv = config_dic[name]['pe']*5
            elif config_dic[name]['simd'] == 16:
                dsp_conv = config_dic[name]['pe']*10
            elif config_dic[name]['simd'] == 32:
                dsp_conv = config_dic[name]['pe']*19
            
            cycle_conv = (config_dic[name]['in_shape'][0]*9)*config_dic[name]['out_shape'][0]*(config_dic[name]['out_shape'][1]*config_dic[name]['out_shape'][2]) / dsp_conv
            

        elif "maxpool" in name :
            # maxpool
            # swu_out
            bram_num += np.ceil(config_dic[pre_name]['out_shape'][0]*8/18)
            # pool_out
            bram_num += np.ceil(config_dic[pre_name]['out_shape'][0]*8/18)
            pre_name = name
            
        elif "avgpool" in name :
            # avgpool
            # swu_out
            bram_num += np.ceil(config_dic[pre_name]['out_shape'][0]*8/18)
            # pool_out
            bram_num += np.ceil(config_dic[pre_name]['out_shape'][0]*8/18)
            pre_name = name

        elif "linear" in name :
            # linear(conv1x1)
            # weight mem
            bram_num += config_dic[name]['pe']* np.ceil(config_dic[name]['simd']*8/18)*np.ceil(((config_dic[name]['in_shape'][0]*9)/config_dic[name]['simd'])*(config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # bias mem
            bram_num += config_dic[name]['pe']* np.ceil(8/18)*np.ceil((config_dic[name]['out_shape'][0]/config_dic[name]['pe'])/1024)
            # padding fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # swu fifo
            bram_num += np.ceil(config_dic[name]['in_shape'][0]*8/18)
            # conv_out fifo
            bram_num += np.ceil(config_dic[name]['out_shape'][0]*8/18)
            pre_name = name
        
        dsp_all += dsp_conv
        cycle_all += cycle_conv
            
    bram_num += np.ceil(64/18)
    
    cycle_all *= 1e-5
    
    return bram_num, dsp_all, cycle_all

def array_to_string(array, elem_bit):
    val = 0
    for i in range(len(array)):
        tmp = array[i]
        tmp2 = tmp
        
        if tmp < 0:
            tmp2 = 2**(elem_bit) + tmp
            
        tmp2 = int(tmp2)
        tmp3 = tmp2 * 2**(elem_bit*i)
        val = val + tmp3
    return val

def pe_simd(np_Vars):
    code = np_Vars[:21]/np.array([8]*5+[1]*16)
    channels = [64,64,128,256,512]
    layers = np_Vars[16:21].astype(int)
    out_channels = []
    for out_ch in range(5):
        out_channels += [(code[out_ch]*channels[out_ch]).astype(int)]*layers[out_ch]
    in_channels = [3]+out_channels[:-1]
    pe = []
    out_ch = 0
    for p in range(5):
        for layer_num in range(int(np_Vars[16+p])):
            while out_channels[out_ch] % (2**int(np_Vars[21+p])) != 0:
                np_Vars[21+p] -= 1
            pe += [2**int(np_Vars[21+p])]
            out_ch += 1
    if int(np_Vars[16])==2:
        simd = [int(np_Vars[26])]+[int(np_Vars[0]/8*64*0.5)]
        in_ch = 2
    else:
        simd = [int(np_Vars[26])]
        in_ch = 1

    for si in range(4):
        for layer_num in range(int(np_Vars[17+si])):
            while in_channels[in_ch] % (2**int(np_Vars[27+si])) != 0:
                np_Vars[27+si] -= 1
            simd += [2**int(np_Vars[27+si])]
            in_ch += 1
    return code,pe,simd

def write2yaml(config,save_path):
    with open(save_path,"w") as f:
        f.write("dataset: "+config['dataset']+'\n')
        f.write("model:\n")
        f.write("  layer_num_max:\n")
        for i in range(5):
            f.write("  - "+str(config['model']['layer_num_max'][i])+'\n')
        f.write("  op_type:\n")
        for i in range(5):
            f.write("  - "+str(config['model']['op_type'][i])+'\n')
        f.write("  pool_type:\n")
        for i in range(6):
            f.write("  - "+str(config['model']['pool_type'][i])+'\n')
        f.write("  stage_layer:\n")
        for i in range(5):
            f.write("  - "+str(config['model']['stage_layer'][i])+'\n')
        f.write("  stage_ratio:\n")
        for i in range(5):
            f.write("  - "+str(config['model']['stage_ratio'][i])+'\n')
        f.write("  with_pool:\n")
        for i in range(len(config['model']['with_pool'])):
            f.write("  - "+str(config['model']['with_pool'][i])+'\n')
        f.write("output:\n")
        f.write("  dir: "+str(config['output']['dir'])+'\n')
        f.write("  epoch_print_freq: "+str(config["output"]['epoch_print_freq'])+'\n')
        f.write("  name: "+"\""+str(config['output']['name'])+"\""+"\n")
        f.write("  print_freq: "+str(config['output']['print_freq'])+'\n')
        f.write("  save_freq: "+str(config['output']['save_freq'])+'\n')
        f.write("train:\n")
        f.write("  batch_size: "+str(config['train']['batch_size'])+'\n')
        f.write("  clip_grad: "+str(config['train']['clip_grad'])+'\n')
        f.write("  ema_alpha: "+str(config['train']['ema_alpha'])+'\n')
        f.write("  ema_update_period: "+str(config['train']['ema_update_period'])+'\n')
        f.write("  epochs: "+str(config['train']['epochs'])+'\n')
        f.write("  label_smoothing: "+str(config['train']['label_smoothing'])+'\n')
        f.write("  loss: "+str(config['train']['loss'])+'\n')
        f.write("  lr_scheduler:\n")
        f.write("    decay_epochs: "+str(config['train']['lr_scheduler']["decay_epochs"])+'\n')
        f.write("    decay_rate: "+str(config['train']['lr_scheduler']["decay_rate"])+'\n')
        # f.write("    min_lr: "+str(config['train']['lr_scheduler']["min_lr"])+'\n')
        f.write("    min_lr: "+str(0.000001)+'\n')
        
        f.write("  no_weight_decay:\n")
        for i in range(len(config['train']['no_weight_decay'])):
            f.write("  - "+str(config['train']['no_weight_decay'][i])+'\n')
        f.write("  optimizer:\n")
        f.write("    base_lr: "+str(config['train']['optimizer']['base_lr'])+'\n')
        f.write("    betas:\n")
        for i in range(len(config['train']['optimizer']['betas'])):
            f.write("    - "+str(config['train']['optimizer']['betas'][i])+'\n')
        f.write("    eps: "+str(config['train']['optimizer']['eps'])+'\n')
        f.write("    momentum: "+str(config['train']['optimizer']['momentum'])+'\n')
        f.write("    name: "+str(config['train']['optimizer']['name'])+'\n')
        # f.write("    repvgg_lr: "+str(config['train']['optimizer']['repvgg_lr'])+'\n')
        f.write("    repvgg_lr: "+str(0.00000001)+'\n')
        
        f.write("    weight_decay_param:\n")
        f.write("      base_decay: "+str(config['train']['optimizer']['weight_decay_param']['base_decay'])+'\n')
        f.write("      echo: "+str(config['train']['optimizer']['weight_decay_param']['echo'])+'\n')
        f.write("      repvgg_decay: "+str(config['train']['optimizer']['weight_decay_param']['repvgg_decay'])+'\n')
        f.write("  start_epoch: "+str(config['train']['start_epoch'])+'\n')
        f.write("  use_l2_norm: "+str(config['train']['use_l2_norm'])+'\n')
        f.write("  warmup_epochs: "+str(config['train']['warmup_epochs'])+'\n')
        f.write("  warmup_lr: "+str(config['train']['warmup_lr'])+'\n')
        f.write("  workers: "+str(config['train']['workers'])+'\n')
        f.write('val:\n')
        f.write("  batch_size: "+str(config['val']['batch_size'])+'\n')
        f.write("  workers: "+str(config['val']['workers'])+'\n')
        
    f.close()
    
