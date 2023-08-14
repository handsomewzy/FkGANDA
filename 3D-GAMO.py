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
parser.add_argument("--n_epochs", type=int, default=5000, help="总共迭代的轮次")
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


class netG_3D(nn.Module):
    def __init__(self, latDim=100, num_class=16, bands=200):
        super(netG_3D, self).__init__()
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


class netD_3D(nn.Module):
    def __init__(self, bands=200, windows=7, numsofclass=16):
        super(netD_3D, self).__init__()
        self.bands = bands
        self.windows = windows
        self.numsofclass = numsofclass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 28, kernel_size=(7, 3, 3), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(28, 28, kernel_size=(7, 3, 3), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(28, 28, kernel_size=(7, 3, 3), padding=(3, 0, 0)),
            nn.BatchNorm3d(28),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(28, 1, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(1),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(bands, 1, 1)),
            nn.Flatten(),
            nn.Sigmoid(),
        )
        # layer5    输出64
        # self.layer6 = nn.Sequential(
        #     nn.Linear(80,1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # labels = labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (100,16,1,1,1)
        # labels = labels.repeat(1, 1, self.bands, self.windows, self.windows)
        # labels = labels * torch.ones_like(x,device = self.device)
        # x = torch.cat([x, labels], dim=1)  # (-1,2,200,7,7)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


if __name__ == "__main__":
    for i in [8,10,15,16]:
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

        # Loss function
        adversarial_loss = torch.nn.MSELoss()

        # Initialize generator and discriminator
        generator = netG_3D(bands=opt.channels)
        discriminator = netD_3D(bands=opt.channels)

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        batches_done = 0
        total_train_step = 0
        for epoch in range(opt.n_epochs):

            for i, (imgs, _) in enumerate(dataloader):
                valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                if cuda:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    real_imgs = real_imgs.cuda()
                    z = z.cuda()

                # Generate a batch of images
                fake_imgs = generator(z).detach()

                # Adversarial loss
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
                loss_D = (real_loss + fake_loss) / 2

                loss_D.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = adversarial_loss(discriminator(gen_imgs), valid)

                loss_G.backward()
                optimizer_G.step()

                batches_done = epoch * len(dataloader) + i
                total_train_step = total_train_step + 1

                print("第 %d 轮训练已经完成！！！d_loss = %f g_loss= %f" % (total_train_step, loss_D, loss_G))

        else:
            print("总共 %d 轮次的训练已经完成！！！" % opt.n_epochs)
            # 模型的存储
            if opt.save_model:
                torch.save(generator.state_dict(), "3DGAMO_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                          opt.training_sample))
