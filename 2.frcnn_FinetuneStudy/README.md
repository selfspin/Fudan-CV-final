## 项目使用指南

### 1. 环境配置
```bash
# 新建虚拟环境
conda create -n mmlab python=3.8 notebook \
pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -y 
# 激活
conda activate mmlab
# 安装mmcv-full 'cu113'和'torch1.11.0'请按自己安装对版本号进行修改
pip install mmcv-full \
-f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
# 安装mmdet
pip install mmdet \
-i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装tensorboard 和 tensorboardX
pip install tensorboard tensorboardX
```

### 2. 数据集
若服务器上没有VOC数据，请先下载
```bash
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
让与项目文件夹同级新建data文件夹，让数据集以data/VOCdevkit/(VOC2007,VOC2012)的结构存放

### 3. 模型效果及eval
详见myDEMO.ipynb


### 4. 模型训练
- 使用COCO上Mask R-CNN的骨干作为预训练权重, 训练日志记录在work_dir中


```bash
nohup \
python tools/train.py MyConfigs/Setting_COCOpretrain.py \
--auto-scale-lr --gpu-id 2 \
>/dev/null  2>&1 &
```
`--gpu-id`: 使用的gpu序号

- 随机权重从头开始训练
```bash
nohup \
python tools/train.py MyConfigs/Setting_init_train.py \
--auto-scale-lr --gpu-id 3 \
>/dev/null  2>&1 &
```

- ImageNet预训练模型赋值给骨干网络训练
```bash
nohup \
python tools/train.py MyConfigs/Setting_ImageNetpretrain.py \
--auto-scale-lr --gpu-id 1 \
>/dev/null  2>&1 &
```

### 5. 可视化训练过程
训练中使用tensorboard可视化
```bash
tensorboard --logdir=work_dirs/
```
