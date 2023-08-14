import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

parser.add_argument("--n_classes", type=int, default=16, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=7, help="和分类网络中的patch size保持一致")

# 数据集设定
parser.add_argument("--channels", type=int, default=200, help="number of image channels"
                                                              "IndianPines 200"
                                                              "PaviaC 102"
                                                              "PaviaU 103"
                                                              "KSC 176"
                                                              "Botswana 145"
                                                              "Houston 144 16"
                                                              "salinas 204 17")

parser.add_argument('--dataset_name', type=str, default='IndianPines', help="使用扩增的数据集名称,如下："
                                                                        "PaviaC"
                                                                        "IndianPines")

parser.add_argument("--dilation", type=int, default=1, help="conv3d hyperparameters")
parser.add_argument("--training_sample", type=float, default=0.05,
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


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
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
    def __init__(self, input_channels):
        dilation = (opt.dilation, 1, 1)
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.channels = input_channels
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(3, 3, 3), dilation=dilation,
                               stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(3, 1, 1), dilation=dilation,
                               stride=(2, 1, 1),
                               padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=(3, 3, 3), dilation=dilation,
                               stride=(1, 1, 1),
                               padding=(1, 0, 0))

        # 为LBN计算图像的大小
        self.size1 = self._get_first_size()
        self.size2 = self._get_second_size()
        self.size3 = self._get_third_size()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            # 针对于WGAN-GP，不能用BN层，因此在鉴别器中使用LayerNorm，加快收敛速度，提高精度。
            # nn.BatchNorm3d(num_features=10, momentum=0.9),
            nn.LayerNorm(self.size1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            # nn.BatchNorm3d(num_features=10, momentum=0.9),
            nn.LayerNorm(self.size2),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=10, out_channels=1, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            # nn.BatchNorm3d(num_features=1, momentum=0.9),
            nn.LayerNorm(self.size3),
            nn.LeakyReLU(),
            nn.Flatten(),

            # IndianPines数据集 145*145*200的特征个数，200个波段，patch*patch*200
            # nn.Linear(2500, 512),

            # PaviaC数据集 1096*492*102，共有102个波段，patch*patch*102
            # nn.Linear(1275, 512),

            # PaviaU数据集 共有103个波段，patch*patch*103
            # nn.Linear(1300, 512),

            # KSC
            # nn.Linear(2200, 512),

            # Botswana
            nn.Linear(1825, 512),

            # Houston
            # nn.Linear(1800, 512),

            # salinas
            # nn.Linear(2550, 512),

            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def _get_first_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channels,
                             7, 7))
            x = self.conv1(x)
            # 展开的大小
            _, t, c, w, h = x.size()
        return [t, c, w, h]

    def _get_second_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channels,
                             7, 7))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return [t, c, w, h]

    def _get_third_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channels,
                             7, 7))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return [t, c, w, h]

    def forward(self, imgs):
        x = self.model(imgs)
        return x


if __name__ == "__main__":
    for i in [2,5,6,12,13,14]:
        opt.choose_labels = [i]
        print(opt.choose_labels)

        # 固定随机数种子，看看效果如何，统一设置为1029
        seed_torch()
        # 数据集的加载
        img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                                      target_folder="Datasets")
        # 训练集和测试集划分,和分类网络保持一致
        train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')

        # 划分为两个训练集，一个用来训练GAN交叉生成样本，一个用来鉴别鉴别器，对半分。
        train_gt1, train_gt2 = utils.sample_gt(train_gt, 0.5, mode='random')

        train_dataset1 = datasets.HyperX_choose(img, train_gt1, dataset=opt.dataset_name, patch_size=7,
                                                choose_labels=opt.choose_labels,
                                                flip_augmentation=False,
                                                radiation_augmentation=False,
                                                mixture_augmentation=False, ignored_labels=[0],
                                                center_pixel=True, supervision='full')
        train_dataset2 = datasets.HyperX_choose(img, train_gt2, dataset=opt.dataset_name, patch_size=7,
                                                choose_labels=opt.choose_labels,
                                                flip_augmentation=False,
                                                radiation_augmentation=False,
                                                mixture_augmentation=False, ignored_labels=[0],
                                                center_pixel=True, supervision='full')

        train_data1_size = len(train_dataset1)
        train_data2_size = len(train_dataset2)
        print("训练集1的长度为：{}".format(train_data1_size))
        print("训练集2的长度为：{}".format(train_data2_size))

        batch_size = min(train_data1_size, train_data2_size)
        dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, drop_last=True)
        dataloader2 = DataLoader(train_dataset2, batch_size=batch_size, drop_last=True)
        dis_dataloader = DataLoader(train_dataset1 + train_dataset2, batch_size=2 * batch_size, drop_last=True)

        # 加载生成器和鉴别器
        generator = Generator()
        # discriminator = Simple_Discriminator()
        discriminator = Discriminator(input_channels=opt.channels)

        # GPU加速
        if cuda:
            generator.cuda()
            discriminator.cuda()

        # Optimizers adam优化器
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # ----------
        #  Training
        # ----------

        # tensorboard 的加载使用
        if opt.tensorboard:
            writer = SummaryWriter("logs")
        total_train_step = 0

        for epoch in range(opt.n_epochs):
            # 首先dataloader1作为训练GAN的数据集，dataloader2作为鉴别器输入。
            # epoch设置为判断标准，两个数据集轮换进行训练,训练5轮次更换一次数据集输入
            if epoch % 5 == 0:
                dataloader1, dataloader2 = dataloader2, dataloader1
            for i, (imgs1, labels) in enumerate(dataloader1):

                batch_size = imgs1.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs1 = Variable(imgs1.type(FloatTensor))

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_imgs = generator(z)

                # 交叉，特征保留生成样本
                p_list = torch.chunk(input=imgs1.type(torch.FloatTensor), chunks=opt.channels, dim=2)
                g_list = torch.chunk(input=gen_imgs.type(torch.FloatTensor), chunks=opt.channels, dim=2)
                mask = compute_mask(p_list, g_list, percent=0.6)
                mixed_imgs = create_mixed_data(p_list, g_list, mask)
                mixed_imgs = Variable(mixed_imgs.type(FloatTensor))

                # 对于鉴别器的假数据
                # if cuda:
                #     gen_imgs.cuda()
                #     mixed_imgs.cuda()

                # 对于鉴别器，两个dataloader的数据视作真数据
                for i, (imgs2, labels2) in enumerate(dataloader2):
                    real_imgs2 = Variable(imgs2.type(FloatTensor))

                # 两部分，生成的图片和交叉得到的图片。二者之和为总的生成器损失。
                validity = discriminator(gen_imgs) + discriminator(mixed_imgs)
                g_loss = -torch.mean(validity)
                if cuda:
                    g_loss.cuda()
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()
                # 鉴别器对于真的判断，由两个数据集的数据组成
                validity_real = discriminator(real_imgs1) + discriminator(real_imgs2)

                # 鉴别器对于假的判断，也由两部分组成，生成的数据和交叉生成的数据
                validity_fake = discriminator(gen_imgs.detach()) + discriminator(mixed_imgs.detach())

                # WGAN-GP鉴别器损失函数表达,使用梯度惩罚。
                gradient_penalty1 = compute_gradient_penalty(discriminator, real_imgs1.data, gen_imgs.data)
                gradient_penalty2 = compute_gradient_penalty(discriminator, real_imgs1.data, mixed_imgs.data)
                gradient_penalty3 = compute_gradient_penalty(discriminator, real_imgs2.data, gen_imgs.data)
                gradient_penalty4 = compute_gradient_penalty(discriminator, real_imgs2.data, mixed_imgs.data)
                gradient_penalty = 0.25 * (
                        gradient_penalty1 + gradient_penalty2 + gradient_penalty3 + gradient_penalty4)
                # gradient_penalty = 0.5 * (gradient_penalty1 + gradient_penalty3)

                d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + 10 * gradient_penalty

                if cuda:
                    d_loss.cuda()

                d_loss.backward()
                optimizer_D.step()

                # print(
                #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                # )

                batches_done = epoch * len(dataloader1) + i
                total_train_step = total_train_step + 1

                print("第 %d 轮训练已经完成！！！d_loss = %f g_loss= %f" % (total_train_step, d_loss, g_loss))

                if opt.tensorboard:
                    if i % 2 == 0:
                        print("训练次数： {} d_loss： {}".format(total_train_step, d_loss))
                        writer.add_scalar("d_loss", d_loss.item(), total_train_step)
                    if i % 2 == 0:
                        print("训练次数： {} g_loss： {}".format(total_train_step, g_loss))
                        writer.add_scalar("g_loss", g_loss.item(), total_train_step)
        else:
            print("总共 %d 轮次的训练已经完成！！！" % opt.n_epochs)

        if opt.tensorboard:
            writer.close()

        # 模型的存储
        if opt.save_model:
            torch.save(generator.state_dict(), "gan_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                     opt.training_sample))
            # 保存鉴别器，用来鉴别交叉比例时不同的选择。
            # torch.save(discriminator.state_dict(), "dis_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
            #                                                              opt.training_sample))
