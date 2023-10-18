# FPGANDA官方代码说明
This is the official source code of the paper 'Features kept generative adversarial network data augmentation strategy for hyperspectral image classification
代码首先训练GAN模型，然后生成新数据后加入分类网络进行训练。

## 安装
安装项目所需的依赖库：

```
pip install -r requirements.txt
```

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
本项目主要基于WGAN，WGAN-GP，CGAN等开源代码开发，baseline等分类网络基于大型的公开HyperX代码，数据集具体的读取方式也与之相同。
