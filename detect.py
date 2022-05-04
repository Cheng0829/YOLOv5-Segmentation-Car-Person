# python train.py --data data/cityscapes_det.yaml --cfg models/yolov5s_city_seg.yaml --batch-size 4 --epochs 100 --workers 1 --label-smoothing 0.1 --img-size 832 --noautoanchor

# python test.py --data data/cityscapes_det.yaml --segdata ./data/citys --weights weights/pspv5m.pt --img-size 1024 --base-size 1024

# python detect.py --weights weights/pspv5m.pt --source data/images/try --conf 0.25 --img-size 1024 --save-as-video

from PIL import Image
import argparse
import time
from pathlib import Path
import os
import cv2
import torch, time
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F

"""****************************************************************************************************************"""
# cjk:只用了model和utils两个文件夹下的py文件
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from PIL import Image

"""****************************************************************************************************************"""

'''
# 1车道,2人行道,3楼房,4未知,5栏杆,
# 6横杆/竖杆,7红绿灯,8路牌,9树,10未知,
# 11天空,12人,13摩托车/电瓶车,14轿车,15未知,
# 16卡车,17公交车,18同13,19自行车
'''

# 1 2 7 8 12 131819 141617
Cityscapes_COLORMAP = [
    [128, 128, 128], [220, 220, 220], [70, 70, 70], [70, 70, 70], [70, 70, 70],
    [70, 70, 70], [99, 192, 133], [254, 156, 144], [70, 70, 70], [70, 70, 70],
    [70, 70, 70], [0, 0, 0], [255, 255, 255], [220, 20, 60], [70, 70, 70],
    [220, 20, 60], [220, 20, 60], [70, 70, 70], [70, 70, 70],
]
# 全255是白色 全0是黑色
Cityscapes_COLORMAP_q = [
    # [70,70,70],[70,70,70],[70,70,70],[70,70,70],[70,70,70],
    # [70,70,70],[70,70,70],[70,70,70],[70,70,70],[70,70,70],
    # [70,70,70],[70,70,70],[70,70,70],[244,35,232],[70,70,70],
    # [244,35,232],[244,35,232],[70,70,70],[70,70,70],
    # 19个:0是白,255是黑
    [128, 64, 128],  # 灰色
    [244, 35, 232],  # 粉色
    [70, 70, 70],  # 深灰色
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],  # person
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

Cityscapes_IDMAP = [[7], [8], [11], [12], [13], [17], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [31],
                    [32], [33]]


# cjk: seg的大小为1 19 480 854 ,经过取max得到两个480X854的矩阵(概率阵和索引阵),
# 相当于用19类的概率去对一张照片比较,然后取每一个地方概率最大的类
# mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Cityscapes_COLORMAP)
def label2image(seg, index, COLORMAP=Cityscapes_COLORMAP):
    # print(len(seg),len(seg[0]),len(seg[0][0]),len(seg[0][0][0])) #->1 19 480 854 #19类 480和854是原始img的size
    # index是索引矩阵
    colormap = np.array(COLORMAP, dtype='uint8')  # 把二维列表改变为二维矩阵形式,其中uint是8位无符号数整型变量
    ##X = pred.astype('int32')# cjk:貌似没什么改变,就删了X
    # print('seg.max(axis=0)*********************************************************************')
    # print(seg.max(axis=0)) #整个tensor内部三维矩阵(立方体)的每一个二维矩阵对应值(面)选一个最大值(概率最大者?),同时返回索引,所以长度为2
    ##seg.max(axis=0)前后两个子矩阵的长度大小都是一样的,都是480行,854列
    # print(seg.max(axis=0)[1])
    # print(len(seg.max(axis=0)[1]),len(seg.max(axis=0)[1][0]))->480 854
    result = colormap[index, :]
    # 二维索引阵的每一个元素索引值对应color中的颜色(三元素列表)
    # print(colormap[index, :])
    # print(len(result),len(result[0]),len(result[0][0])) #-> 480 854 3
    return result


def trainid2id(pred, IDMAP=Cityscapes_IDMAP):
    colormap = np.array(IDMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]


def frame_photo_video(path, size):
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    fps = 30
    size = (1920, 1080)  # 图片的分辨率片
    i = 1
    while True:
        if os.path.isdir('./runs/detect/' + 'exp-video{}.avi'.format(i)) == True:
            i = i + 1
        else:
            break
    file_path = './runs/detect/' + 'exp-video{}.avi'.format(i)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for item in filelist:
        if item.endswith('.jpg'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频


def video_frame_photo(path):
    print('视频预处理中!')
    save_path = 'data\\images\\video\\' + os.path.basename(path).split('.')[0]
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        count = count + 1
        cv2.imencode('.jpg', image)[1].tofile(save_path + r"\frame%d.jpg" % count)
        success, image = vidcap.read()
    print('视频预处理完毕!')
    return save_path


def detect(set_model, live_img=''):
    if (set_model == 1):
        source_try = 'data/images/try'  # 图片detect
    elif (set_model == 2):
        print('请输入视频路径:')
        path = input()
        source_try = video_frame_photo(path)  # 视频detect
    elif (set_model == 3):
        source_try = live_img  # 实时detect
    else:
        raise AttributeError

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/pspv5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source_try, help='source')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    parser.add_argument('--submit', action='store_true', help='get submit file in folder submit')
    opt = parser.parse_args()
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size  # 命令行参数传递
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    mark_classes = ['road', 'sidewalk', 'traffic light', 'traffic sign', 'person', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle']
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # print(device) #-> cuda:0
    half = device.type != 'cpu'  # half precision only supported on CUDA 原始代码,cpu用float32,gpu用float16
    # Load model
    """**********************************************************************************************************************************"""
    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride   # =32
    # print(stride) #-> 32
    # imgsz = check_img_size(imgsz, s=stride)  #
    if half:
        # print(imgsz)
        # print('half-model**********************************************************************************************')
        model.half()  # to FP16  
        # -> 改精度:FP32是单精度浮点数,用8bit表示指数,23bit表示小数.FP16是半精度浮点数,可以省内存
        # print(model)

    # Set Dataloader
    vid_path, vid_writer, s_writer = None, None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # 执行else语句
        cudnn.benchmark = False
        dataset = LoadImages(source_try, 1024, 32)

    if opt.submit or opt.save_as_video:  # 提交和做视频必定是同尺寸
        cudnn.benchmark = True

    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # 共10种颜色,每种颜色里面的三个灰度值随机

    # Run inference
    if device.type != 'cpu':
        # print('cpu-model**********************************************************************************************')
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    flag = 1
    #
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)  # from_numpy()使np.ndarray->torch.tensor,to(device)是指定设备(GPU)运行

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加一个维度

        # Inference
        with torch.no_grad():
            t1 = time_synchronized()
            out = model(img, None)
            pred = out[0][0]
            seg = out[1]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # 
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # 此时det与pred完全一样
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # 输出图片分辨率
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 更换前4列,即坐标变换
                # Print results
                # 识别结果并输出字符串
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # n为某一类别的总数
                    s += "{} {}{}, ".format(n, names[int(c)], "s" * (n > 1))  # n>1则输出复数s
                    # 1024x768 2 traffic signs,4 persons,1 bus,1 train,

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if 'car' in label:
                            pass  # continue
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.5f}s)')
            seg = F.interpolate(seg, (im0.shape[0], im0.shape[1]), mode='bilinear',
                                align_corners=True)  # 采样函数,# 第二个参数size——输出空间大小,第三个线性插值。

            seg = seg[0]
            mask = label2image(seg, seg.max(axis=0)[1].cpu().numpy(), Cityscapes_COLORMAP)[:, :,
                   ::-1]  # 不加[:, :, ::-1]也行,但是颜色对比度没那么大
            # a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。
            # cv2.imshow()
            dst = cv2.addWeighted(im0, 0.8, mask, 0.2, 0)
            # return dst
            if set_model == 3:
                # cv2.imshow(str(p), im0)
                cv2.imshow("segmentation", mask)
                cv2.imshow("mix", dst)
                cv2.waitKey(1)  # 1 millisecond
            # save_img = 0
            if set_model == 1:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path[:-4] + "_mask" + save_path[-4:], mask)
                    cv2.imwrite(save_path[:-4] + "_dst" + save_path[-4:], dst)

            if set_model == 2:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path[:-4] + "_dst" + save_path[-4:], dst)
                    # print(save_path[:-4])
    if set_model == 2:
        frame_photo_video(('\\').join(save_path[:-4].split('\\')[:-1]), (1920, 1080))

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if s_writer != None:
        s_writer.release()
    print(f'Done. ({time.time() - t0:.3f}s)')
