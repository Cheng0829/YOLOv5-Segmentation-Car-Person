# common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
import torch.nn.functional as F


class RFB2(nn.Module):  # 目标检测
    # 魔改模块,除了历史遗留(改完训练模型精度不错，不想改名重训)名字叫RFB，其实和RFB没啥关系了(参考deeplabv3的反面级联结构，也有点像CSP，由于是级联，d设置参考论文HDC避免网格效应)实验效果不错，能满足较好非线性、扩大感受野、多尺度融合的初衷(在bise中单个精度和多个其他模块组合差不多，速度和C3相近比ASPP之类的快)
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[2, 3], has_globel=False):
        # 第一个3*3的d相当于1，典型的设置1,2,3; 1,2,5; 1,3,5
        super(RFB2, self).__init__()
        self.out_channels = out_planes
        self.has_globel = has_globel
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1),
            Conv(inter_planes, inter_planes, k=3, s=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1),
        )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(inter_planes, inter_planes, k=1),
            )
        self.ConvLinear = Conv(int(5 * inter_planes) if has_globel else int(4 * inter_planes), out_planes, k=1, s=1)

    def forward(self, x):  # 思路就是rate逐渐递进的空洞卷积连续卷扩大感受野避免使用rate太大的卷积(级联注意rate要满足HDC公式且不应该有非1公倍数，空洞卷积网格效应)，多个并联获取多尺度特征
        x3 = self.branch3(x)  # １＊１是独立的　类似C3，区别在于全部都会cat
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        if not self.has_globel:
            out = self.ConvLinear(torch.cat([x0, x1, x2, x3], 1))
        else:
            x4 = F.interpolate(self.branch4(x2), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0, x1, x2, x3, x4], 1))
        return out


class PyramidPooling(nn.Module):  # 语义分割:PSPNet: Pyramid Scene Parsing Network
    def __init__(self, in_channels, k=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(k[0])
        self.pool2 = nn.AdaptiveAvgPool2d(k[1])
        self.pool3 = nn.AdaptiveAvgPool2d(k[2])
        self.pool4 = nn.AdaptiveAvgPool2d(k[3])
        #直接按照论文模型进行编写,比较简单
        # AdaptiveAvgPool2d（二元自适应均值池化层）
        # 二元: 二维矩阵
        #
        out_channels = in_channels // 4  # 整除,256/4=64
        self.conv1 = Conv(in_channels, out_channels, k=1)
        self.conv2 = Conv(in_channels, out_channels, k=1)
        self.conv3 = Conv(in_channels, out_channels, k=1)
        self.conv4 = Conv(in_channels, out_channels, k=1)

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)
        # torch.cat()是将两个张量tensor拼接在一起，cat是concatenate的意思，即拼接
        return torch.cat((x, feat1, feat2, feat3, feat4,), 1)


class FFM(nn.Module):  # 特征融合:FeatureFusionModule  reduction用来控制瓶颈结构,upsample
    def __init__(self, in_chan, out_chan, reduction=1, is_cat=True, k=1):
        super(FFM, self).__init__()
        self.convblk = Conv(in_chan, out_chan, k=k, s=1, p=None)  ## 注意力处用了１＊１瓶颈，两个卷积都不带bn,一个带普通激活，一个sigmoid
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(out_chan, out_chan // reduction,
                                                         kernel_size=1, stride=1, padding=0, bias=False),
                                               nn.SiLU(inplace=True),
                                               nn.Conv2d(out_chan // reduction, out_chan,
                                                         kernel_size=1, stride=1, padding=0, bias=False),
                                               nn.Sigmoid(),
                                               )
        self.is_cat = is_cat

    def forward(self, fspfcp):  # 空间, 语义两个张量用[]包裹送入模块，为了方便Sequential
        fcat = torch.cat(fspfcp, dim=1) if self.is_cat else fspfcp
        feat = self.convblk(fcat)
        atten = self.channel_attention(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


#
class Conv(nn.Module):
    # Standard convolution  通用卷积模块,包括1卷积1BN1激活,激活默认SiLU,可用变量指定,不激活时用nn.Identity()占位,直接返回输入
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)

        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # SiLU是一个非线性激活函数:SiLU(x)=x/[1+e^(-x)]
        # x = [1.0, 2.0, 3.0, 4.0] -> nn.silu(x) = [0.731059, 1.761594, 2.857722, 3.928055]

    # detect不执行
    def forward(self, x):
        # print(777)
        return self.act(self.bn(self.conv(x)))

    # detect执行
    def fuseforward(self, x):
        # print(888)
        return self.act(self.conv(x))


#
# autopad为same卷积或者same池化自动扩充
# 通过卷积核的大小来计算需要的padding为多少才能把tensor补成原来的形状
def autopad(k, p=None):  # kernel, padding  # 自动padding,不指定p时自动按kernel大小pading到same
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad  # k为可迭代对象时,支持同时计算多个pading量
    return p


#
class Bottleneck(nn.Module):
    # Standard bottleneck 残差块
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):  # 如果shortcut并且输入输出通道相同则跳层相加
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


#
class C3(nn.Module):  # 5.0版本模型backbone和head用的都是这个,V5用C3替代了CSP
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # n个残差组件(Bottleneck)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


#
class SPP(nn.Module):  # 目标检测
    # Spatial pyramid pooling layer used in YOLOv3-SPP # ModuleLis容器多分支实现SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 输入卷一次
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 输出卷一次(输入通道:SPP的len(k)个尺度cat后加输入)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


#
class Focus(nn.Module):  # 卷积复杂度O(W*H*C_in*C_out)此操作使WH减半,后续C_in翻4倍, 把宽高信息整合到通道维度上,
    # Focus wh information into c-space  # 相同下采样条件下计算量会减小,　后面Contract, Expand用不同的方法实现同样的目的
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # return self.conv(self.contract(x))
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


#
class Concat(nn.Module):  # 用nn.Module包装了cat方法
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
