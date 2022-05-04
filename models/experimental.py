# YOLOv5 experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.google_utils import attempt_download


class Ensemble(nn.ModuleList):
    # print('cjk-test-ex:114')
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        # print('cjk-test-ex:120')
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):  # 和ultralytics一样,加载权重
    # model = attempt_load(weights, map_location=device)# device==cuda:0
    # print('cjk-test-ex:131')
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    # 译:加载一组权重=[a,b,c]的模型,或单个权重=[a]或a的模型
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:  # 加载权重
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    # #比ultra少了个 new Detect Layer compatibility
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:  #
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # 比ultra在for里多了个'stride',其他一样
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
