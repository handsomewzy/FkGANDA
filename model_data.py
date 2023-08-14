import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from torchvision import datasets
from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import DataLoader
import datasets
import utils
import wgan3D

# 和生成器参数保持一致
opt = wgan3D.opt
img_shape = wgan3D.img_shape


# cuda = True if torch.cuda.is_available() else False
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 三维反卷积生成新数据
        self.model = nn.Sequential(
            nn.ConvTranspose3d(1, 10, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(10, momentum=0.9),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(10, 10, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(10, momentum=0.9),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(10, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(1, momentum=0.9),
            nn.LeakyReLU(),

            nn.Flatten(),

            # nn.Linear(1791, 1024),
            # nn.Linear(1024, int(np.prod(img_shape))),
            # nn.Tanh(),

            nn.Linear(1791, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

        # 简单的全连接生成新数据，最后展成指定数据大小
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.simple = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        # 复杂的三维卷积的生成
        gen_img = x
        gen_img = gen_img.unsqueeze(1)
        gen_img = gen_img.unsqueeze(-1)
        gen_img = gen_img.unsqueeze(-1)
        img = self.model(gen_img)
        img = img.view(img.size(0), *img_shape)

        # 简单的FNN生成
        # img = self.simple(x)
        # img = img.view(img.shape[0], *img_shape)

        return img


class Simple_Discriminator(nn.Module):
    def __init__(self):
        super(Simple_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Discriminator(nn.Module):
    def __init__(self):
        dilation = (opt.dilation, 1, 1)
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=10, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=10, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=10, out_channels=1, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=1, momentum=0.9),
            nn.LeakyReLU(),
            nn.Flatten(),

            # IndianPines数据集 145*145*200的特征个数，200个波段，patch*patch*200
            # nn.Linear(2500, 1024),

            # PaviaC数据集 1096*492*102，共有102个波段，patch*patch*102
            # nn.Linear(1275, 1024),

            # PaviaU数据集 共有103个波段，patch*patch*103
            # nn.Linear(1300, 1024),

            # KSC
            # nn.Linear(2200, 1024),

            # Botswana
            # nn.Linear(1825, 1024),

            # Houston
            nn.Linear(1800, 512),

            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, imgs):
        x = self.model(imgs)
        return x


img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                              target_folder="Datasets")
train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')


def create_data(label, pth, num=1, img=img, gt=train_gt, data_name=opt.dataset_name, copy=False, gan=False):
    """
    用来选择生成数据的函数，可以选择生成交叉数据，纯复制，或者GAN生成的。
    :param label: 需要生成数据的标签信息
    :param pth: 预先训练好的GAN的地址
    :param num: 扩增的数目
    :param img: 原始的图片数据
    :param gt: 原始的标签信息
    :param data_name: 扩增的数据集名称
    :param copy: 是否进行copy操作，bool值类型，默认为False
    :param gan: 是否进行纯GAN生成操作，bool值类型，默认为False
    :return:
    """
    train_dataset = datasets.HyperX_choose(img, gt, patch_size=7, choose_labels=[label],
                                           center_pixel=True, supervision='full')
    train_dataloader = DataLoader(train_dataset, batch_size=num, drop_last=True)

    # load the model
    Gen_model = Generator()
    Gen_model.load_state_dict(torch.load(pth))
    # generate the image
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (num, 100))))
    gen_labels = Variable(torch.LongTensor(np.random.choice([label], num)))
    gen_imgs = Gen_model(z)
    for imgs, targets in train_dataloader:
        # 5波段交叉,按照评价指标方差进行交叉保留
        p1, p2, p3, p4, p5 = torch.chunk(input=imgs, chunks=5, dim=2)
        g1, g2, g3, g4, g5 = torch.chunk(input=gen_imgs, chunks=5, dim=2)

        # IndianPines数据集样本1 7 9的扩增
        if data_name == "IndianPines":
            if label == 1:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # IP数据集样本1 034 10011
                    new = torch.cat([p1, g2, g3, p4, p5], 2)
            elif label == 7:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # IP数据集样本7 023 10110
                    new = torch.cat([p1, g2, p3, p4, g5], 2)
            elif label == 9:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # IP数据集样本9 034 10011
                    new = torch.cat([p1, g2, g3, p4, p5], 2)
            break

        elif data_name == "PaviaC":
            # 使用波段选择出交叉的片段然后处理数据返回数据集，扩增的样本为3 4 7
            if label == 2:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # 034 10011
                    new = torch.cat([p1, g2, g3, p4, p5], 2)

            elif label == 3:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # 034 10011
                    new = torch.cat([p1, g2, g3, p4, p5], 2)

            elif label == 4:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # 012 11100
                    new = torch.cat([p1, p2, p3, g4, g5], 2)

            elif label == 5:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # 123 01110
                    new = torch.cat([g1, p2, p3, p4, g5], 2)

            elif label == 7:
                if copy:
                    new = torch.cat([p1, p2, p3, p4, p5], 2)
                elif gan:
                    new = torch.cat([g1, g2, g3, g4, g5], 2)
                else:
                    # 034 10011
                    new = torch.cat([p1, g2, g3, p4, p5], 2)
            break

    NewDataset = datasets.Create_dataset(gen_imgs=new, gen_labels=gen_labels)
    return NewDataset


class WGAN_Generator(nn.Module):
    def __init__(self):
        super(WGAN_Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class WGAN_Discriminator(nn.Module):
    def __init__(self):
        super(WGAN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class DCGAN_Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(3072, int(np.prod(img_shape))),
        )

        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        x = x.view(x.shape[0], *img_shape)
        return x


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


class _3DGAMO(nn.Module):
    def __init__(self, latDim=100, num_class=16, bands=200):
        super(_3DGAMO, self).__init__()
        self.num_class = num_class
        self.latDim = latDim
        self.bands = bands
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(self.latDim, self.bands, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                               padding=(0, 0, 0)),
            nn.BatchNorm3d(self.bands),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(1, 28, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(28, 1, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.Tanh()
        )

    def forward(self, input):
        x = input
        x = x.view(x.shape[0], self.latDim, 1, 1, 1)
        out = self.layer1(x)
        out = out.view(out.shape[0], 1, self.bands, 1, 1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class ACWGANGP(nn.Module):
    def __init__(self):
        super(ACWGANGP, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = 2  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Flatten(),
            # IP
            # nn.Linear(12800, int(np.prod(img_shape))),
            # PU
            # nn.Linear(6592, int(np.prod(img_shape))),
            # houston
            # nn.Linear(9216, int(np.prod(img_shape))),
            # salina
            nn.Linear(13056, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = img.view(img.shape[0], *img_shape)
        return img
# 样本1 7 9的数据扩增 IP数据集
# MyDataset_1 = model_data.create_data(1, pth="gan_model1.pth", num=2, img=img, gt=train_gt, data_name='IndianPines')
# MyDataset_7 = model_data.create_data(7, pth="gan_model_7wgan.pth", num=2, img=img, gt=train_gt, data_name='IndianPines')
# MyDataset_9 = model_data.create_data(9, pth="gan_model9.pth", num=1, img=img, gt=train_gt, data_name='IndianPines')

# 样本3 7 8的数据扩增，PaviaC数据集
# MyDataset_3 = create_data(3, pth="gan_3_PaviaC.pth", num=8, img=img, gt=train_gt, data_name="PaviaC",
#                                      copy=False, gan=False)
# MyDataset_4 = create_data(4, pth="gan_4_PaviaC.pth", num=7, img=img, gt=train_gt, data_name="PaviaC",
#                                      copy=False, gan=False)
# MyDataset_7 = create_data(7, pth="gan_7_PaviaC.pth", num=18, img=img, gt=train_gt, data_name="PaviaC",
#                                      copy=False, gan=False)
