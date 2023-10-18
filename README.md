# FPGANDA官方代码说明
This is the official source code of the paper 'Features kept generative adversarial network data augmentation strategy for hyperspectral image classification'

代码首先训练GAN模型，然后生成新数据后加入分类网络进行训练。

论文链接：[https://www.sciencedirect.com/science/article/abs/pii/S0031320323003990]

## 安装
正常的torch环境，按照DeepHyperX配置即可。
参考链接：[https://github.com/nshaud/DeepHyperX]

## 使用说明
### 训练GAN
配置好数据集路径信息后，运行如下代码：

```
python keepGAN.py
```

### 基于生成结果进行波段选择
训练好GAN生成样本后，进行波段选择。

```
python Band_Select.py
```

### 基于扩增数据训练分类器
根据选择到的波段，融合生成新的扩增数据加入网络进行训练，具体的细节参数配置参考Completed_Band_Select.py文件

```
python Completed_Band_Select.py
```


## 说明
本项目主要基于WGAN，WGAN-GP，CGAN等开源代码开发，baseline等分类网络基于大型的公开DeepHyperX代码，数据集具体的读取方式也与之相同。

