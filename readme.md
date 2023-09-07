# YoloV8 使用文档

## 安装

### 安装显卡驱动

```
1、通过设备管理器查看显卡型号， 文档使用的显卡型号是(NVIDIA GeForce RTX 3050 Ti Laptop GPU)
2、下载显卡驱动 https://www.nvidia.cn/download/index.aspx?lang=cn
3、选择对应的先看驱动，下载类型选择 (Studio 驱动程序 (SD))
4、搜索后下载，然后一直下一步直到完成
5、安装完成以后通过 cmd 命令 输入 nvidia-smi 查看显卡驱动信息以及版本号信息
```

### 安装 CUDA

建议使用 11.7， 以减少兼容性问题

```
1、下载地址： https://developer.nvidia.com/cuda-toolkit-archive , 找到对应的版本(nvidia-smi)
2、选择自定义安装， 全选安装内容， 然后 Next
3、安装成功后检测是否有对应的环境变量（以下出现的版本号与自己下载的对应）
CUDA_PATH           C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
CUDA_PATH_V11_7     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
```

### 安装 cudnn

- 下载 cudnn 需要 注册一个 NVIDIA account
- 下载的 cudnn 需要和 CUDA 的版本号对应

```
https://developer.nvidia.com/cudnn
然后将下载的压缩包解压后复制 bin, include, lib 三个文件夹复制到 cuda 的安装目录 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7)
然后通过CMD 执行命名 nvcc -V 测试是否成功， 以下是成功后的提示
如果提示命令无效， 需要重启电脑

C:\Users\dayin>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:59:34_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

### 安装 py

https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 安装 yolov8

```
pip install ultralytics
```

## 训练

### 准备数据与配置

- 在任意的位置创建项目文件夹如 yolov8
- 在项目文件夹中创建 datasets 文件夹
- 在 datasets 创建训练配置文件， 如本项目中的 coco128.yaml
- 把训练数据的文件夹放在 datasets 文件夹下， 如本项目的中 datasets/coco128

配置文件说明如下

```
path： 数据集根文件夹， 实际可以填绝对路径
train: 训练数据集， 相对于数据集根目录的路径， 实际可以填绝对路径
val: 验证数据集， 相对于数据集根目录的路径， 实际可以填绝对路径
test: 测试集， 相对于数据集根目录的路径， 实际可以填绝对路径
names: 因 labels 使用的是数字， 此映射会识别结果显示的文字映射
```

### 训练

### Windows

```

yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
```

### Mac

```
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
```

### 参数说明

https://docs.ultralytics.com/modes/train/#multi-gpu-training

#### model

```
model=yolov8n.pt 是指预训练模式， 按场景选择， 越大的模型效果越好， 但训练和使用开销变大， 如果在使用在移动设备，
或边缘设备， 则需要按需调整。

https://docs.ultralytics.com/models/yolov8/#supported-modes

YOLOv8n、YOLOv8s、YOLOv8m、YOLOv8l、YOLOv8x
```

#### epochs

迭代次数， 一般先跑一个 100， 看训练结果会不会过拟合， 然后逐步上调。

#### batch

一次训练多少张图片， 可以先设置为 -1， 会自动检测合适的数字， 后续就固定使用那个数字。和图片大小、显存大小有关。
尽可能设置大。

#### imgsz

图片训练大小

#### cache

图片缓存到显存， 建议开启， 可以极大的加快训练速度。若不开启， 图片会从图片、显存大量的交换。

#### device

使用什么设置进行训练。cpu/gpu
