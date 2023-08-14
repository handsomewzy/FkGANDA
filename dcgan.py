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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(torch.nn.Module):
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


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
        # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Flatten(),
            nn.Sigmoid())
        self.l1 = nn.Linear(int(np.prod(img_shape)), 3072)
        self.f1 = nn.Flatten()

    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        x = x.view(x.shape[0], *(3, 32, 32))
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)

class Simple_Discriminator(nn.Module):
    def __init__(self):
        super(Simple_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

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
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        generator = Generator(channels=3)
        discriminator = Discriminator(channels=3)
        # discriminator = Simple_Discriminator()
        print(cuda)

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
        for epoch in range(20000):
            for i, (imgs, _) in enumerate(dataloader):
                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(torch.FloatTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim, 1, 1))))

                if cuda:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    real_imgs = real_imgs.cuda()
                    z = z.cuda()

                # Generate a batch of images
                gen_imgs = generator(z)
                # print(gen_imgs.shape)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                # print(discriminator(gen_imgs))
                # print(discriminator(real_imgs))

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2


                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                batches_done = epoch * len(dataloader) + i
                total_train_step = total_train_step + 1

                print("第 %d 轮训练已经完成！！！d_loss = %f g_loss= %f real_loss= %f fake_loss= %f" % (
                total_train_step, d_loss, g_loss, real_loss, fake_loss))

        else:
            print("总共 %d 轮次的训练已经完成！！！" % opt.n_epochs)
            # 模型的存储
            if opt.save_model:
                torch.save(generator.state_dict(), "DCGAN_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                           opt.training_sample))
