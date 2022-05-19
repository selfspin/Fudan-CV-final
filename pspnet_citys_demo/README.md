这是一个cityscape预训练PSPNet的测试demo

#### 环境

mxnet/mxnet-gpu

gluoncv

matplotlib, numpy

#### 数据

测试视频、分割视频、每帧图像数据可以从云盘获取：

链接：https://pan.baidu.com/s/1dScYvFc2exmNshyKiwNBhw 
提取码：efil

测试视频来源：

https://www.pexels.com/video/person-driving-in-a-city-street-under-a-blue-sky-4483549/

#### 测试

下载数据后，运行main.py对每帧图像进行测试并将分割结果保存至mask文件夹下

运行mask.py将原始图像与分割图像叠加并保存至out文件夹下

图像合成视频使用了win10自带的视频编辑器

#### demo

<video src="drive+mask.mp4"></video>

