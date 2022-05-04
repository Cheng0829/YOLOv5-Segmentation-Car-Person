import cv2,os
import numpy as np
from PIL import Image
from detect import detect

if __name__ == "__main__":

    if (a==1):
        print('o')

    capture = cv2.VideoCapture(0)
    i = 0
    print("请选择功能模式:\n1.图片检测\n2.视频检测\n3.摄像头实时监测")
    print("请输入:")
    model = eval(input())
    if model == 1 or model == 2:
        detect(model)
    while model == 3:
        # 读取某一帧
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        # frame = np.array(yolo.detect_image(frame))
        frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        save = 'live/{}.jpg'.format(i)
        f = open(save, 'a')
        cv2.imwrite(save, frame)
        i = i + 1
        c = detect(3, save)
