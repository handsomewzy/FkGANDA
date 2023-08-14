# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init

# utils
import math
import os
import datetime
import numpy as np
import joblib
import argparse
import os
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datasets
import utils
from Completed_Band_Select import compute_mask, create_mixed_data

# ——————————————————————
# 生成器一些参数的设置 opt
# ——————————————————————
os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20000, help="总共迭代的轮次")
parser.add_argument("--batch_size", type=int, default=16, help="批度，对于小样本一次全部加入"
                                                               "后续由训练集长度替代")
parser.add_argument("--lr_gen", type=float, default=0.00005, help="learning rate of the generator")
parser.add_argument("--lr_dis", type=float, default=0.00005, help="learning rate of the discriminator")

# Adam优化器的参数设置，改用RMSprop后不需要，wgan思路
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

parser.add_argument("--n_classes", type=int, default=17, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=7, help="和分类网络中的patch size保持一致")

# 数据集设定
parser.add_argument("--channels", type=int, default=204, help="number of image channels"
                                                              "IndianPines 200"
                                                              "PaviaC 102"
                                                              "PaviaU 103"
                                                              "KSC 176"
                                                              "Botswana 145"
                                                              "Houston 144 16"
                                                              "salinas 204 17")

parser.add_argument('--dataset_name', type=str, default='salinas', help="使用扩增的数据集名称,如下："
                                                                       "PaviaC"
                                                                       "IndianPines")

parser.add_argument("--dilation", type=int, default=1, help="conv3d hyperparameters")
parser.add_argument("--training_sample", type=float, default=0.01,
                    help="Percentage of samples to use for training (default: 10%)"
                         "应该和扩增网络的训练集的参数保持一致")
parser.add_argument("--choose_labels", default=[1], help="想要扩增的样本，一次选择一个，一次训练一个样本类别的网络")

parser.add_argument('--save_model', type=str, default=True, help="是否选择保存训练的模型")
parser.add_argument('--tensorboard', type=str, default=False, help="是否使用tensorboard记录训练数据")

opt = parser.parse_args()
print(opt)

img_shape = (1, opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

cuda = True if torch.cuda.is_available() else False


def seed_torch(seed=1029):
    """
    设置随机数种子，保证每次的结果大差不差可以复现。使得每次运行结果类似，波动不再那么巨大。
    :param seed:随机数种子的设定
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HSGAN_Generator(nn.Module):
    """
    Semisupervised Hyperspectral Image Classification Based on Generative Adversarial Networks
    Ying Zhan , Dan Hu, Yuntao Wang
    http://www.ieee.org/publications_standards/publications/rights/index.html
    """

    def __init__(self, batch_size):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with
        # sigmoid gate activation and PRetanh activation functions for hidden representations
        super(HSGAN_Generator, self).__init__()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(100, 1024)

        self.fc2 = nn.Linear(1024, 6400)
        self.fc2_bn = nn.BatchNorm1d(1)

        self.up1 = nn.Upsample(size=100)
        self.up1_bn = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(128, 64, 1)
        self.up2 = nn.Upsample(size=200)
        self.conv2 = nn.Conv1d(64, 1, 1)

        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(200, int(np.prod(img_shape)))
        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(100, 1024),
        #     nn.Linear(1024, 6400),
        # )
        #
        # self.conv_blocks = nn.Sequential(
        #     nn.Upsample(size=100),
        #     nn.Conv1d(128, 64, 1),
        #     nn.Upsample(size=200),
        #     nn.Conv1d(64, 1, 1)
        # )

    # shape of x is batch size*1*100,它默认的为100*1*100
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2_bn(self.fc2(x)))
        x = x.view(self.batch_size, 128, 50)
        x = self.up1_bn(self.up1(x))
        x = torch.tanh(self.conv1(x))
        x = self.up2(x)
        x = self.conv2(x)
        # x = self.linear_blocks(x)
        # x = x.view(100, 128, 50)
        # x = self.conv_blocks(x)
        x = self.f1(x)
        x = self.l1(x)
        x = x.view(x.shape[0], *img_shape)

        return x


class HSGAN_Discriminator(nn.Module):
    """
    Semisupervised Hyperspectral Image Classification Based on Generative Adversarial Networks
    Ying Zhan , Dan Hu, Yuntao Wang
    http://www.ieee.org/publications_standards/publications/rights/index.html
    """

    # input_channels=200 n_classes=17
    def __init__(self, batch_size):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64
        # with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(HSGAN_Discriminator, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.conv2 = nn.Conv1d(32, 32, 3)

        # self.conv_blocks1 = nn.Sequential(
        #     nn.Conv1d(1, 32, 1),
        #     nn.Conv1d(32, 32, 3),
        # )
        self.max_pooling = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(32, 32, 3)
        self.linear_block = nn.Linear(1504, 1024)
        # real or fake

        self.real_or_fake = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        # class

        self.classify = nn.Sequential(
            nn.Linear(1024, 17),
            nn.Softmax(dim=1),
        )
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(int(np.prod(img_shape)), 200)

    def forward(self, x):  # (batch size, 1, 200)
        x = self.f1(x)
        x = self.l1(x)
        x = x.view(x.shape[0], *(1, 200))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.max_pooling(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv3(x))
        x = self.max_pooling(x)
        x = x.view(self.batch_size, 1, 1504)
        x = self.linear_block(x)
        # real or fake
        validity = self.real_or_fake(x)
        # classify
        label = self.classify(x)
        return validity, label


if __name__ == "__main__":
    for i in [16]:
        opt.choose_labels = [i]
        print(opt.choose_labels)

        # 固定随机数种子，看看效果如何，统一设置为1029
        seed_torch()
        # 数据集的加载
        img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                                      target_folder="Datasets")
        # 训练集和测试集划分,和分类网络保持一致
        train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')
        print("训练集大小: {} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        train_dataset = datasets.HyperX_choose(img, train_gt, dataset=opt.dataset_name, patch_size=7,
                                               choose_labels=opt.choose_labels,
                                               flip_augmentation=False,
                                               radiation_augmentation=False,
                                               mixture_augmentation=False, ignored_labels=[0],
                                               center_pixel=True, supervision='full')
        train_data_size = len(train_dataset)
        print("训练集的长度为：{}".format(train_data_size))
        dataloader = DataLoader(train_dataset, batch_size=train_data_size, drop_last=True)

        # Initialize generator and discriminator
        generator = HSGAN_Generator(batch_size=train_data_size)
        discriminator = HSGAN_Discriminator(batch_size=train_data_size)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Loss functions
        adversarial_loss = torch.nn.BCELoss()
        auxiliary_loss = torch.nn.CrossEntropyLoss()

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
            auxiliary_loss.cuda()

        # ----------
        #  Training
        # ----------

        batches_done = 0
        total_train_step = 0
        for epoch in range(opt.n_epochs):

            for i, (imgs, _) in enumerate(dataloader):
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                labels = _
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_D.zero_grad()

                valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                gen_labels = Variable(torch.FloatTensor(imgs.shape[0], ).fill_(0.0), requires_grad=False)
                gen_labels = gen_labels.to(torch.int64)

                # Sample noise as generator input
                # HSGAN的噪声 z 大小为（batch size，1,100）
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 1, opt.latent_dim))))

                if cuda:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    real_imgs = real_imgs.cuda()
                    labels = labels.cuda()
                    z = z.cuda()
                    gen_labels = gen_labels.cuda()

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # print(fake_imgs.shape)

                validity, pred_label = discriminator(fake_imgs)
                validity = validity.view(imgs.shape[0], 1)
                pred_label = pred_label.view(imgs.shape[0], 17)
                # Adversarial loss
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
                g_loss.backward()
                optimizer_G.step()

                # -----------------
                #  Train Discriminator
                # -----------------
                optimizer_D.zero_grad()

                output, real_aux = discriminator(real_imgs)

                output = output.view(imgs.shape[0], 1)
                # real_aux = real_aux.view(imgs.shape[0], 16)
                real_aux = real_aux.view(imgs.shape[0], 17)

                d_real_loss = (adversarial_loss(output, valid) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(fake_imgs.detach())

                fake_pred = fake_pred.view(imgs.shape[0], 1)
                # fake_aux = fake_aux.view(imgs.shape[0], 16)
                fake_aux = fake_aux.view(imgs.shape[0], 17)

                d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                total_train_step = total_train_step + 1
                print("第 %d 轮训练已经完成！！！d_loss = %f g_loss= %f" % (total_train_step, d_loss, g_loss))

        else:
            print("总共 %d 轮次的训练已经完成！！！" % opt.n_epochs)
            # 模型的存储
            if opt.save_model:
                torch.save(generator.state_dict(), "HSGAN_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                          opt.training_sample))
