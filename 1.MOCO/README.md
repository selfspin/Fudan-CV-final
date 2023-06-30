## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


### Unsupervised Training

使用 MoCo v1:
```
python main_moco.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
使用 MoCo v2, 设置 `--mlp --moco-t 0.2 --aug-plus --cos`


### Linear Classification

使用上述训练好的模型进行下游任务微调（以cifar10，训练全网络为例）：
```
python main_lincls.py \
  -a resnet18 \
  --lr 30.0 \
  --lr-b 1e-5 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0100.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --cifar10
```

### Models

