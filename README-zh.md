# 多功能检测与分割系统
 
[Introduction to English Version](https://github.com/Cheng0829/yolov5-segmentation-car-person/blob/master/README.md)

## 概述
 
**这是我的本科毕业设计**

它的主要功能是通过YOLOv5进行目标检测，并使用PSPNet进行语义分割。
本项目YOLOv5部分代码基于 *[ultralytics YOLO V5 tag v5.0](https://github.com/ultralytics/yolov5)* 。
相应地，我也使用了ultralytics/YOLOv5的预训练模型。
我通常使用两个最简单的预训练模型--yolov5s.pt和yolov5s.pt。你可以在./weights中直接看到它们。
在语义分割部分，我使用了PSPNet（全称为 *[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)*
，即金字塔场景解析网络，此网络模型于2017年在CVPR上提出。
在我看来，这是性能和简洁性之间平衡得最好的网络之一。
事实上，在我拿到YOLOv5源代码后，我只花了一点时间就完成主要部分，而我花了大部分剩余时间将此模块与YOLO结合使用。

## 演示

### 典型的一组处理结果

![](demo_image/38.png)
![](demo_image/39.png)
![](demo_image/40.png)

## 文件和文件夹

### ./weights

该文件夹包含四个训练前文件。yolov5s.pt和yolov5m.pt是目标检测模型、pspv5s.pt和pspv5m.pt是语义分割模型。它们都有很好的性能，但由于我的显卡性能不足，分割模型仍然有一些缺点。
需要指出的是，“xx5m.pt”的性能要好于“xx5s.pt”，因为它的网络更加复杂。

### ./runs

此文件夹包含检测、测试或训练项目的结果。

### ./demo_Image

此文件夹包含一些演示图像。

### ./data

此文件夹包含所有数据集。 我主要通过CityScapes数据集训练我的分割模型，并在目标检测部分采用了预训练模型。
当然，如果您想继续训练YOLOv5模型，您将使用Microsoft Coco数据集。

### ./models

这个文件夹对我来说是最重要的。
在Common.py中，我主要添加了三个模块，RFB2和FFM起辅助作用，金字塔池模块起到分割作用，这部分代码都是仿照 *[PSPNet](https://arxiv.org/abs/1612.01105)* 论文模型编写的。
当然，有很多模块来自于ultralytics/YOLOv5，我删除了所有未使用的模块。
在yolo.py中，我添加了一个名为SegMaskPSP的类，它集成了Common.py中提到的三个模块，当训练模型时，parse_model函数将调用它。

### train.py

该python文件与ultralytics/YOLOv5中的train.py文件大致相同。当我研究YOLO算法的时候，我对其做了一些更改。
它看起来有很大变化，但主体结构没有改变。
您可以使用cmd代码
`python Train.py--data ciyscapes_de.yaml--cfg yolov5s_City_Seg.yaml--批量大小18--纪元200--权重。/yolov5s.pt--工人8--标签平滑0.1--img-大小832--非自动锚定`
。
训练你自己的模型
当然，如果显卡的性能不是很高，您可以尝试降低workers运算单元的数量和batch-size的大小。
当然，我建议你直接使用我的预训练模型，或者如果你不缺钱的话，可以去找一些云GPU平台(事实上，我已经去过一个叫AutoDL的网站很多次了)。

### test.py

通过一些著名的指标，如准确率、召回率、F1等来测试您的模型的性能。
您可以使用cmd代码。
`python test.py--data data/Cityscapes_Det.yaml--Segdata./data/citys--Weights/pspv5m.pt--img-size1024--base-size1024`。
来测试你的模型。

### Detect.py

如果你只想直接使用我的项目，这个文件对你来说可能是最重要的。
在文件中，您可以选择三种模式：①图像检测、②视频检测 和 ③摄像头实时检测。

### predict.py

该python文件将调用detect.py进行检测
**其他文件夹或文件来自ultralytics/YOLOv5，基本上无关紧要。**。

### requirements s.txt

您可以使用`pip install -r requirements s.txt`安装所有需要的包。
如果你使用的是Windows系统，你应该把txt文件中的‘pycotools’替换为‘pycoTools-windows’。
当然，我猜您在安装它们时会遇到一些错误。我建议您使用Conda来创建一个新的Python虚拟环境。安装Anaconda后，您可以启动“Anaconda PowerShell Prompt”。
创建虚拟环境的代码是`conda create-n you_env_name python=x.x` .
您也可以使用`conda install-n you_env_name package_name`来安装其他包。