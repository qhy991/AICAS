from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .RepVGGBlock import RepVGGBlock


def make_layer(stage_num, layer_num, channel_num_in, channel_num_out, op_type,
               with_pool,pool_type):
    channel_nums_in = [channel_num_in] + [channel_num_out] * (layer_num - 1)
    layers = []
    if stage_num == 0 :
        first_layer_stride = 1
    else:
        first_layer_stride = 2
    if with_pool == True:
        if pool_type == "avgpool":
            layers.append(("avgpool", nn.AvgPool2d(2, 2)))
        else:
            layers.append(("maxpool", nn.MaxPool2d(2, 2)))
        if op_type == 'vgg':
            layers.append(("stage_{}_0_vgg".format(stage_num),
                           VGGBlock(channel_num_in,
                                    channel_num_out,
                                    kernel_size=3,
                                    stride=1)))
            layers += [("stage_{}_{}_vgg".format(stage_num, i),
                        VGGBlock(channel_num_out, channel_num_out, 3))
                       for i in range(1, layer_num)]
        else:
            layers.append(("stage_{}_0_repvgg".format(stage_num),
                           RepVGGBlock(channel_num_in,
                                       channel_num_out,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)))
            layers += [("stage_{}_{}_repvgg".format(stage_num, i),
                        RepVGGBlock(channel_num_out,
                                    channel_num_out,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)) for i in range(1, layer_num)]

    else:
        if op_type == 'vgg':
            layers.append(("stage_{}_0_vgg".format(stage_num),
                           VGGBlock(channel_num_in,
                                    channel_num_out,
                                    kernel_size=3,
                                    stride=first_layer_stride)))
            layers += [("stage_{}_{}_vgg".format(stage_num, i),
                        VGGBlock(channel_num_out, channel_num_out, 3))
                       for i in range(1, layer_num)]
        else:
            layers.append(("stage_{}_0_repvgg".format(stage_num),
                           RepVGGBlock(channel_num_in,
                                       channel_num_out,
                                       kernel_size=3,
                                       padding=1,
                                       stride=first_layer_stride)))
            layers += [("stage_{}_{}_repvgg".format(stage_num, i),
                        RepVGGBlock(channel_num_out,
                                    channel_num_out,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)) for i in range(1, layer_num)]
    return nn.Sequential(OrderedDict(layers))
def VGGBlock(in_channels,
             out_channels,
             kernel_size,
             stride=1,
             padding=1,
             dilation=1,
             groups=1,
             padding_mode='zeros'):
    conv2d = nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size=kernel_size,
                       stride = stride,
                       padding=1,
                       dilation=1,
                       groups=1,
                       padding_mode='zeros')
    layers = nn.Sequential(
        OrderedDict([("conv", conv2d), ("bn", nn.BatchNorm2d(out_channels)),
                     ("relu", nn.ReLU(inplace=True))]))
    return layers

class Net_VWW(nn.Module):
    def __init__(self,config, num_classes):
        super(Net_VWW, self).__init__()
        self.pool = False
        self.stage_0 = make_layer(0, 1, 3, 64, "vgg",
               with_pool=False,pool_type="None")
        # self.pool = torch.nn.MaxPool2d(2,2)
        self.stage_1 = make_layer(1, 1, 64, 16, "vgg",
               with_pool=False,pool_type="None")
        self.stage_2 = make_layer(2, 1, 16, 1, "vgg",
               with_pool=False,pool_type="None")
        
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.linear = nn.Linear(256, num_classes)
        
    def forward(self, input):
        out = self.stage_0(input)
        # out = self.pool(out)
        out = self.stage_1(out)
        # out = self.pool(out)
        out = self.stage_2(out)
        # out = self.gap(out)
        # print(out.shape)
        out = self.linear(out.view(out.size(0), -1))
        # out = self.classifier(out.view(out.size(0), -1))
        return out