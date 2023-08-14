import argparse
import os
import numpy as np
import math

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datasets
import utils
import torch.autograd as autograd


# ——————————————————————
# 生成器一些参数的设置 opt
# ——————————————————————
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=19, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.0002, help="adam: learning rate of the generator")
parser.add_argument("--lr_dis", type=float, default=0.0002, help="adam: learning rate of the discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=17, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=7, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=200, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--dilation", type=int, default=1, help="conv3d hyperparameters")
parser.add_argument("--training_sample", type=float, default=0.2,
                    help="Percentage of samples to use for training (default: 10%)")
parser.add_argument("--choose_labels", default=[1, 7, 9], help="the target we choose to generate")
parser.add_argument('--dataset', type=str, default='IndianPines', help="Dataset to use.")
opt = parser.parse_args()
print(opt)

img_shape = (1, opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        #
        # self.model = nn.Sequential(
        #     *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(img_shape))),
        #     nn.Tanh()
        # )
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1, out_channels=10, kernel_size=(2, 2, 2), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
            nn.BatchNorm3d(num_features=10),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(in_channels=10, out_channels=20, kernel_size=(2, 2, 2), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
            nn.BatchNorm3d(num_features=20),
            nn.LeakyReLU(),

            nn.MaxPool3d(kernel_size=(3, 3, 3)),

            nn.ConvTranspose3d(in_channels=20, out_channels=10, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
            nn.BatchNorm3d(num_features=10),
            nn.LeakyReLU(),

            nn.Flatten(),
            *block(370, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_img = torch.cat((self.label_emb(labels), noise), -1)
        gen_img = gen_img.unsqueeze(1)
        gen_img = gen_img.unsqueeze(-1)
        gen_img = gen_img.unsqueeze(-1)
        img = self.model(gen_img)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        dilation = (opt.dilation, 1, 1)
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.l1 = nn.Linear(6517, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 16)
        self.l4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=20),
            # nn.LeakyReLU(),

            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=20),
            # nn.LeakyReLU(),

            nn.Conv3d(in_channels=20, out_channels=35, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=35),
            # nn.LeakyReLU(),

            nn.Conv3d(in_channels=35, out_channels=35, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=35),
            # nn.LeakyReLU(),

            nn.Conv3d(in_channels=35, out_channels=20, kernel_size=(3, 1, 1), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=20),
            # nn.LeakyReLU(),

            nn.Conv3d(in_channels=20, out_channels=10, kernel_size=(2, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=10),
            # nn.LeakyReLU(),

            nn.Flatten(),
            # nn.Linear(910,512),
            # nn.Linear(512,256),
            # nn.Linear(256,16),
            # nn.Linear(16,1)
        )

    def forward(self, imgs, target):
        x = self.model(imgs)
        x = torch.cat((x, self.label_embedding(target)), -1)
        x = F.leaky_relu(self.l1(x))
        # x = self.dropout(x)
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = F.sigmoid(x)
        return x

# -------------------------------------------------
# BGAN损失函数，加入是否能改进结果
# -------------------------------------------------
def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)


# Configure data loader ip数据集的加载
img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name='IndianPines',
                                                                              target_folder="./dataset")

dataset = datasets.HyperX(img, gt, patch_size=opt.img_size, ignored_labels=ignored_value, flip_augmentation=False,
                          radiation_augmentation=False, mixture_augmentation=False, center_pixel=True,
                          supervision='full',dataset=opt.dataset)

dataloader = DataLoader(dataset, batch_size=16, drop_last=True)

# 训练集和测试集划分
train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')
print("Training size: {} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))

# 训练dataloader
train_dataset = datasets.HyperX_choose(img, train_gt, patch_size=opt.img_size, choose_labels=opt.choose_labels,
                                       center_pixel=True, supervision='full')
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True)
dataloader = train_dataloader
train_data_size = len(train_dataset)
print("训练集的长度为：{}".format(train_data_size))

# 鉴别器数据尺寸尝试
# Dis = Discriminator()
# for imgs,target in dataloader:
#     x = Dis(imgs,target)
#     print(x.shape)

# 生成器数据尺寸尝试
# Gen = Generator()
# for imgs, target in dataloader:
#     z = Variable(torch.FloatTensor(np.random.normal(0, 1, (16, opt.latent_dim))))
#     gen_labels = Variable(torch.LongTensor(np.random.randint(0, opt.n_classes, 16)))
#     gen_img = Gen(z, gen_labels)
#     print(gen_img.shape)

# Loss weight for gradient penalty
lambda_gp = 10

# ----------
#  计算梯度惩罚，WGAN-GP
# ----------
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates,target=1)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_gen, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_dis, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# tensorboard 的加载使用
writer = SummaryWriter("logs")
total_train_step = 0

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        gen_labels = Variable(LongTensor(np.random.choice(opt.choose_labels, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)


        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)

        # g_loss = -torch.mean(validity)
        g_loss = adversarial_loss(validity, valid)
        # g_loss = boundary_seeking_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # --------------------------------------------------------------
        # WGAN-GP鉴别器损失函数表达
        # --------------------------------------------------------------
        # d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

        # --------------------------------------------------------------
        # 为防止d-loss趋近于0，当d-loss太小时，不对鉴别器进行梯度更新，不进行训练，优化结果
        # --------------------------------------------------------------
        if d_loss > 0:
            d_loss.backward()
            optimizer_D.step()
        # d_loss.backward()
        # optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        batches_done = epoch * len(dataloader) + i
        total_train_step = total_train_step + 1

        if i % 2 == 0:
            print("训练次数： {} d_loss： {}".format(total_train_step, d_loss))
            writer.add_scalar("d_loss", d_loss.item(), total_train_step)
        if i % 2 == 0:
            print("训练次数： {} g_loss： {}".format(total_train_step, g_loss))
            writer.add_scalar("g_loss", g_loss.item(), total_train_step)

        # 保存训练图片
        # if batches_done % opt.sample_interval == 0:
        #     sample_image(n_row=16, batches_done=batches_done)
writer.close()

# 模型的存储
torch.save(generator.state_dict(), "gan_model.pth")
torch.save(discriminator.state_dict(), "dis_model.pth")
