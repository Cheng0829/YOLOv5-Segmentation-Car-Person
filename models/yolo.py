# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
import torch.nn as nn
import onnx.external_data_helper

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


# test使用forward函数
# detect使用forward函数
class SegMaskPSP(nn.Module):  # 框架来自于https://www.bilibili.com/video/BV1FT4y1E74V?p=138,
    # print('cjk-test-yolo:161')
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):
        # n是C3的, (接口保留了,没有使用)c_hid是隐藏层输出通道数（注意配置文件s*0.5,m*0.75,l*1）
        # print('cjk-test-yolo:163')
        super(SegMaskPSP, self).__init__()
        # 用16,19,22宁可在融合处加深耗费一些时间，检测会涨点分割也很好。严格的消融实验证明用17,20,23分割可能还会微涨，但检测会掉３个点以上，所有头如此
        self.c_in8 = ch[0]  # 16  
        self.c_in16 = ch[1]  # 19
        self.c_in32 = ch[2]  # 22
        # self.c_aux = ch[0]  # 辅助损失  找不到合适地方放辅助，放弃
        self.c_out = n_segcls
        # 注意配置文件通道写256,此时s模型c_hid＝128
        # torch.nn.Sequential():这是一个有顺序的容器，将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行
        # 当module使用SegMaskPSP时,依次通过RFB2,PyramidPooling,FFM,Conv2d等神经网络模块的处理
        self.out = nn.Sequential(

            RFB2(c_hid * 3, c_hid, d=[2, 3], map_reduce=6),  # 和RFB-Net无关,仅仅是训练时忘了改名
            # PyramidPooling前应加入非线性强一点的层并适当扩大感受野
            # 所以一般检测网络anchor的大小的获取都要依赖不同层的特征图，因为不同层次的特征图，其感受野大小不同，这样检测网络才会适应不同尺寸的目标。
            PyramidPooling(c_hid, k=[1, 2, 3, 6]),  # 1236是PSP官方给出的
            FFM(c_hid * 2, c_hid, k=3, is_cat=False),
            nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 上采样
        )
        self.m8 = nn.Sequential(
            Conv(self.c_in8, c_hid, k=1),
        )
        self.m32 = nn.Sequential(
            Conv(self.c_in32, c_hid, k=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.m16 = nn.Sequential(
            Conv(self.c_in16, c_hid, k=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        # print('cjk-test-yolo:196')
        # 这个头三层融合输入做过消融实验，单独16:72.6三层融合:73.5,建议所有用1/8的头都采用三层融合，在Lab的实验显示三层融合的1/16输入也有增长
        feat = torch.cat([self.m8(x[0]), self.m16(x[1]), self.m32(x[2])], 1)
        # return self.out(feat) if not self.training else [self.out(feat), self.aux(x[0])]
        return self.out(feat)


# test使用forward函数
# detect使用forward和_make_grid函数
class Detect(nn.Module):  # 检测头
    # print('cjk-test-yolo:204')
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        # print('cjk-test-yolo:209')
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 每个anchor输出通道=nc类别+1是否有目标+4放缩偏移量
        self.nl = len(anchors)  # number of detection layers anchors是列表的列表,外层几个列表表示有几个层用于输出
        self.na = len(anchors[0]) // 2  # number of anchors  内层列表表示该层anchor形状尺寸,//即该层anchor数
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2) anchor参数是模型非计算图参数,用register_buffer保存(buffer parameter)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 三个输出层输入通道不一样

    # gird和输出特征图一样大,值对应此anchor中心, anchor_grid张量也同尺寸, 两个值对应了此anchor的尺寸
    def forward(self, x):
        # print('cjk-test-yolo:223')
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):  # 分别对三个输出层处理
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 输出x[i]变形BCHW(C=na*no) --> B,na,H,W,no(由第二维区分三个anchor),  no=nc+5,  x是3个张量的列表, 一个张量表一个输出层
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()  # 所有通道输出sigmoid, 后1+类别数通道自然表示有无目标和目标种类, 前4个通道按公式偏移放缩anchor
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy 中心偏移公式见issue
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 大小放缩公式见issue
                z.append(y.view(bs, -1, self.no))  # 0输入时保证0偏移, 中心0输入0.5输出,偏到grid中心(yolo anchor从左上角算起))
        # 训练直接返回变形后的x去求损失, 推理对                                    # 大小0输入1输出,乘以anchor尺寸不变, 公式限制最大放大倍数为4倍
        return x if self.training else (
            torch.cat(z, 1), x)  # 注意训练模式和测试(以及推理)模式不同, 训练模式仅返回变形后的x, 测试推理返回放缩偏移后的box(即z)和变形后x

    @staticmethod
    def _make_grid(nx=20, ny=20):  # 用来生成anchor中心(特征图每个像素下标即其anchor中心)的函数
        # print('cjk-test-yolo:246')
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# test会使用model类下的fuse,info,两个forward函数
# detect会使用model类下的fuse,fuse,info,两个forward函数
class Model(nn.Module):  # 核心模型
    # print('cjk-test-yolo:252')
    def __init__(self, cfg='yolov5m.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        # print('cjk-test-yolo:254')
        super(Model, self).__init__()
        if isinstance(cfg, dict):  # 配置可直接接收字典
            self.yaml = cfg  # model dict
        else:  # is *.yaml  更多是用yaml解析配置
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, 'rb') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 字典的get方法,配置文件有ch就把模型输入通道配成ch,没有就按默认值ch=3
        if nc and nc != self.yaml['nc']:  # 若Model类初始化指定了nc(非None)且和配置文件不等,以Model类初始化值为准,并修改字典值
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:  # 若Model类初始化指定了anchor值,以Model类初始化为准,并修改字典值
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        '''*****************************************************************************************************'''
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist 解析配置文件
        '''*****************************************************************************************************'''

        self.save.append(24)  # 增加记录分割层
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()  Detect头
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(2, ch, s, s))[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)  # 初始化, 看代码只初始化了BN和激活函数,跳过了卷积层
        self.info()
        logger.info('')

    # 与ultra一样
    def forward(self, x, augment=False, profile=False):
        # print('cjk-test-yolo:294')
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    # 除了y[-2],其他和ultra一样
    def forward_once(self, x, profile=False):
        # print('cjk-test-yolo:315')
        y = []  # outputs  用于记录中间输出的y
        for m in self.model:
            if m.f != -1:  # if not from previous layer 非单纯上一层则需要调整此层输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # 输入来自单层, 直接取那层输出           来自多层, 其中-1取输入x, 非-1取那层输出
            # 调好输入每层都是直接跑, detect是最后一层, for循环最后一个自然是detect结果
            x = m(x)  # run
            y.append(
                x if m.i in self.save else None)  # save output 解析时self.save记录了需要保存的那些层(后续层输入用到),仅保存这些层输出即可(改版代码新增记录分割层24)

        return [x, y[-2]]  # 检测, 分割

    # 与ultra相比把Conv+DWConv换成了单独的Conv
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # print('cjk-test-yolo:364')
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    # 和ultra一样
    def info(self, verbose=False, img_size=640):  # print model information
        # print('cjk-test-yolo:397')
        model_info(self, verbose, img_size)

    # 和ultra一样
    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # 译:将偏差初始化为Detect(),cf属于频数类
        # print('cjk-test-yolo:341')
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# parse_model即分析模型
# train使用
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw, n_segcls = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['n_segcls']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) yolo输出通道数 = anchor数 * (类别+1个是否有目标+4个偏移放缩量)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings 执行字符串表达式,block名转函数/类,字符数字转数字
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 同上,
            except:
                pass
        # n控制深度, yaml配置文件中num为1就1次,num>1就 num*depth_multiple次, 即此block本身以及block子结构重复次数
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, Focus, C3]:
            c1, c2 = ch[f], args[0]  # 指定层输入(c1)输出(c2)通道数(ch记录各层输出通道,f表输入层下标,输入层的输出通道就是本层输入通道)
            if c2 != no:  # if not output 对非输出层, 原作者此处代码有风险
                c2 = make_divisible(c2 * gw, 8)  # 实际输出通道数是 配置文件的c2 * width_multiple 并向上取到可被8整除

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # number of repeats 对C3和BottleneckCSP来说深度n代表残差模块的个数, C3TR的n表transformer的head数
                n = 1  # 置1表示深度对这三个模块是控制子结构重复, 而不是本身重复
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # 对BN层, 参数就是输入层的通道数
        elif m is Concat:
            c2 = sum([ch[x] for x in f])  # Concat层, 输出通道就是几个输入层通道数相加
        elif m is Detect:
            args.append([ch[x] for x in f])  # 检测层, 把来源下标列表f中的层输出通道数加入args中, 用于构建Detect的卷积输入通道数
            if isinstance(args[1], int):  # number of anchors 一般跑不进这句, args[1]是anchors在配置文件中已用列表写好, 非int
                args[1] = [list(range(args[1] * 2))] * len(f)
            ############"************************************************************************************************"
            "************************************************************************************************"
        elif m in [SegMaskPSP]:  # 语义分割头
            args[1] = max(round(args[1] * gd), 1) if args[1] > 1 else args[1]  # SegMaskPSP中C3的n(Lab里用来控制ASPP砍多少通道)
            args[2] = make_divisible(args[2] * gw, 8)  # SegMask C3(或其他可放缩子结构) 的输出通道数
            args.append([ch[x] for x in f])
            # n = 1 不用设1了, SegMask自己配置文件的n永远1
            "************************************************************************************************"
        ############"************************************************************************************************"
        # elif m is Contract:
        #    c2 = ch[f] * args[0] ** 2
        # elif m is Expand:
        #    c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(
            *args)  # module 深度控制C3等的block子结构重复次数(见上if中n置为1), 对Conv等则是其本身重复次数
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 由来源记哪些层的结果保存
        layers.append(m_)  # 解析结果加到layers列表
        if i == 0:
            ch = []  # 如果第一层,新建ch列表保存输出通道数
        ch.append(c2)  # 保存此层输出通道数, 下一层输入通道
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5m_city_seg.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
