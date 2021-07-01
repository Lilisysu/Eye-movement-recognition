# 眼动识别

## Background

随着电子产品的普及，屏幕使用变多，判断使用者凝视的屏幕位置有重要的价值。当人眼看向屏幕不同位置时，眼球相对于屏幕摄像头的位置以及眼睛的形态会有不同，基于以上两点，利用神经网络训练出一个模型，通过摄像头获取使用者的照片，获得使用者观察的屏幕位置。

## Install

1. 使用环境为

   - python 3.6

   - Keras 2.1.5

   - tensorflow 1.14.0

2. 从压缩包解压所有文件
3.权重下载链接：https://pan.baidu.com/s/1brQ19SV61eMlIqpkeL1Haw 
提取码：0000 
--来自百度网盘超级会员V1的分享
将四个权重放到logs/000文件夹下

## Usage

1. 人眼检测（获得眼睛框）

   终端运行 python test.py --xx 

   其中xx类别与对应功能如下：

   xx=image 检测单张图片中的人眼

   xx=input 视频检测人眼

   ​   input=0 使用摄像头检测人眼

   ​	  input=path  检测已经录好的视频，其中path是存放路径

   xx=boxes 检测图片生成的检测框，保存在boxes.TXT

   xx=image_batch_test：批量检测图片，在图片上绘制检测框，保存至result文件夹

2. 眼球所观察位置检测

   运行 Real_time_Position.py

## File

- VOC：存放训练所需的数据

- logs：模型权重

- model_data：模型参数

- Location_Image：测试图片的存放位置

- result：检测结果存放路径
- yolo3：网络模型
- boxes.TXT 检测框结果


## Notice

1. 测试时只会识别有两只眼睛的图片，且脸靠近屏幕的测试效果较好，脸部正对屏幕较佳；
2. 测试时利用了多帧图片的结果求均值，同时也有一定的时延，因此眼球缓慢移动效果较好；