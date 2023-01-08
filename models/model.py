from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .RepVGGBlock import RepVGGBlock


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


def make_layer(stage_num, layer_num, channel_num_in, channel_num_out, op_type,
               with_maxpool,pool_type):
    channel_nums_in = [channel_num_in] + [channel_num_out] * (layer_num - 1)
    layers = []
    if stage_num == 0 :
        first_layer_stride = 1
    else:
        first_layer_stride = 2
    if with_maxpool == True:
        if pool_type == "avgpool":
            layers.append(("maxpool", nn.AvgPool2d(2, 2)))
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


class Net(nn.Module):
    def __init__(self, config, num_classes):
        super(Net, self).__init__()
        self.pool = False
        self.stage_0 = make_layer(
            0, config["model"]["stage_layer"][0], 3,
            int(config["model"]["layer_num_max"][0] *
                config["model"]["stage_ratio"][0]),
            config["model"]["op_type"][0], config["model"]["with_pool"][0],
            config["model"]["pool_type"][0])
        self.stage_1 = make_layer(
            1, config["model"]["stage_layer"][1],
            int(config["model"]["layer_num_max"][0] *
                config["model"]["stage_ratio"][0]),
            int(config["model"]["layer_num_max"][1] *
                config["model"]["stage_ratio"][1]),
            config["model"]["op_type"][1], config["model"]["with_pool"][1],
            config["model"]["pool_type"][1])
        self.stage_2 = make_layer(
            2, config["model"]["stage_layer"][2],
            int(config["model"]["layer_num_max"][1] *
                config["model"]["stage_ratio"][1]),
            int(config["model"]["layer_num_max"][2] *
                config["model"]["stage_ratio"][2]),
            config["model"]["op_type"][2], config["model"]["with_pool"][2],
            config["model"]["pool_type"][2])
        self.stage_3 = make_layer(
            3, config["model"]["stage_layer"][3],
            int(config["model"]["layer_num_max"][2] *
                config["model"]["stage_ratio"][2]),
            int(config["model"]["layer_num_max"][3] *
                config["model"]["stage_ratio"][3]),
            config["model"]["op_type"][3], config["model"]["with_pool"][3],
            config["model"]["pool_type"][3])
        self.stage_4 = make_layer(
            4, config["model"]["stage_layer"][4],
            int(config["model"]["layer_num_max"][3] *
                config["model"]["stage_ratio"][3]),
            int(config["model"]["layer_num_max"][4] *
                config["model"]["stage_ratio"][4]),
            config["model"]["op_type"][4], config["model"]["with_pool"][4],
            config["model"]["pool_type"][4])
        if config["model"]['with_last_pool']:
            self.pool = True
            if config["model"]["pool_type"] == "maxpool":
                self.pool = nn.MaxPool2d(2,2)
            else:
                self.pool = nn.AvgPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.last_pool = last_pool
        # self.linear = nn.Linear(
        #     int(config["model"]["layer_num_max"][4] *
        #         config["model"]["stage_ratio"][4]), int(config["model"]["layer_num_max"][4] *
        #         config["model"]["stage_ratio"][4]))
        self.linear = nn.Linear(
            int(config["model"]["layer_num_max"][4] *
                config["model"]["stage_ratio"][4]), num_classes)
    def forward(self, input):
        out = self.stage_0(input)
        num = 1
        for stage in (self.stage_1, self.stage_2, self.stage_3, self.stage_4):
            for block in stage:
                out = block(out)
        # print(out.shape)
        if self.pool:
            out = self.pool(out)
        out = self.gap(out)
        out = self.linear(out.view(out.size(0), -1))
        # out = self.linear2(out)
        
        
        return out
