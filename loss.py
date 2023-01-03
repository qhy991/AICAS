import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math


class BCELoss(_Loss):
    def __init__(self, loss_weight=1.0, bias=False):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight
        self.bias = bias

    def forward(self, input, label):
        C = input.size(1) if self.bias else 1
        if label.dim() == 1:
            one_hot = torch.zeros_like(input).cuda()
            label = label.reshape(one_hot.shape[0], 1)
            one_hot.scatter_(1, label, 1)
            loss = self.bce_loss(input, one_hot)
        elif label.dim() > 1:
            label = label.float()
            loss = self.bce_loss(input - math.log(C), label) * C
        return loss.mean() * self.loss_weight


class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(min=self.eps)
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        # import pdb
        # pdb.set_trace()
        return loss * self.loss_weight

# <<<<<<< HEAD
    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)
# =======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial



class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=88,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        # logger = get_root_logger()
        print(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)
# >>>>>>> bd3bee7309cad09bd8bfc7a79d531e33e9e7a08e

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

# <<<<<<< HEAD
# =======
    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w
# >>>>>>> bd3bee7309cad09bd8bfc7a79d531e33e9e7a08e
import numpy as np
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list)) # 常系数 C
        m_list = torch.cuda.FloatTensor(m_list) # 转成 tensor
        self.m_list = m_list
        assert s > 0
        self.s = s # 这个参数的作用论文里提过么？
        self.weight = weight # 和频率相关的 re-weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8) # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1) # dim idx input
        index_float = index.type(torch.cuda.FloatTensor) # 转 tensor
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m # y 的 logit 减去 margin
        output = torch.where(index, x_m, x) # 按照修改位置合并
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    
"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
# import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self):
        super(BalancedSoftmax, self).__init__()
        # with open(freq_path, 'r') as fd:
        #     freq = json.load(fd)
        freq = [229, 291, 372, 447, 825, 113, 1170, 444, 940, 271, 582, 322, 421, 298, 227, 35, 180, 184, 410, 189, 249, 153, 191, 307, 310, 301, 73, 78, 110, 78, 250, 76, 135, 102, 87, 73, 118, 111, 377, 219, 165, 324, 257, 1725, 658, 23, 82, 458, 325, 145, 480, 502, 108, 259, 601, 234, 16, 269, 108, 2102, 304, 588, 1335, 1360, 385, 363, 162, 293, 13, 16, 3069, 1152, 1189, 1184, 1186, 2592, 29, 172, 169, 171, 1274, 8, 2346, 2345, 2344, 5121, 34, 460, 150]
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


# def create_loss():
#     print('Loading Balanced Softmax Loss.')
#     return BalancedSoftmax()
