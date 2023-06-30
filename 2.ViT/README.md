# Vision Transformer

## 环境配置

git clone并配置python环境

```shell
git clone https://github.com/selfspin/Fudan-CV-final
cd ViT
conda create -n vit python=3.8
conda activate vit
pip install -r requirements.txt
```

## 训练

使用默认参数训练

```shell
python train_vit.py --device 0 1 2 3
python train_resnet.py --device 0 1 2 3
```

可更改参数可以查看帮助

```shell
python train_vit.py -h
python train_resnet.py -h
```

## 测试

模型参数保存在`model`文件夹下

测试CIFAR100上的Top1准确率

```shell
python test.py --device 0 1 2 3
```

测试16张图上的预测

```shell
python visualize.py
```

使用PyCharm等IDE运行以看到可视化图片

