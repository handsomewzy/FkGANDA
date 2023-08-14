import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datasets
import utils

# ——————————————————————
# 生成器一些参数的设置 opt
# ——————————————————————
os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="总共迭代的轮次")
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
parser.add_argument("--channels", type=int, default=200, help="number of image channels"
                                                              "IndianPines 200"
                                                              "PaviaC 102"
                                                              "PaviaU 103"
                                                              "KSC 176"
                                                              "Botswana 145")

parser.add_argument('--dataset_name', type=str, default='IndianPines', help="使用扩增的数据集名称,如下："
                                                                         "PaviaC"
                                                                         "IndianPines")

parser.add_argument("--dilation", type=int, default=1, help="conv3d hyperparameters")
parser.add_argument("--training_sample", type=float, default=0.05,
                    help="Percentage of samples to use for training (default: 10%)"
                         "应该和扩增网络的训练集的参数保持一致")
parser.add_argument("--choose_labels", default=[12], help="想要扩增的样本，一次选择一个，一次训练一个样本类别的网络")

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
        self.model = nn.Sequential(
            nn.ConvTranspose3d(1, 10, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(10),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(10, 10, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(10),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(10, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(1791, 2048),

            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        gen_img = x
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
        self.l1 = nn.Linear(2500, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 16)
        self.l4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=10),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=10),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=10, out_channels=1, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=1),
            nn.LeakyReLU(),

            # nn.Conv3d(in_channels=35, out_channels=35, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
            #           padding=(1, 0, 0)),
            # nn.BatchNorm3d(num_features=35),
            # nn.LeakyReLU(),
            #
            # nn.Conv3d(in_channels=35, out_channels=20, kernel_size=(3, 1, 1), dilation=dilation, stride=(1, 1, 1),
            #           padding=(1, 0, 0)),
            # nn.BatchNorm3d(num_features=20),
            # nn.LeakyReLU(),
            #
            # nn.Conv3d(in_channels=20, out_channels=10, kernel_size=(2, 1, 1), dilation=dilation, stride=(2, 1, 1),
            #           padding=(1, 0, 0)),
            # nn.BatchNorm3d(num_features=10),
            # nn.LeakyReLU(),
            nn.Flatten(),

            # IndianPines数据集 145*145*200的特征个数，200个波段，patch*patch*200
            # nn.Linear(2500, 1024),

            # PaviaC数据集 1096*492*102，共有102个波段，patch*patch*102
            # nn.Linear(1275, 1024),

            # PaviaU数据集 共有103个波段，patch*patch*103
            # nn.Linear(1300, 1024),

            # KSC
            # nn.Linear(2200,1024),

            # Botswana
            nn.Linear(1825, 1024),

            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, imgs):
        x = self.model(imgs)
        return x


if __name__ == "__main__":
    # 固定随机数种子，看看效果如何，统一设置为1029
    seed_torch()
    # 数据集的加载
    img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                                  target_folder="Datasets")
    # 训练集和测试集划分,和分类网络保持一致
    train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')
    print("训练集大小: {} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    train_dataset = datasets.HyperX_choose(img, train_gt, patch_size=opt.img_size, choose_labels=opt.choose_labels,
                                           center_pixel=True, supervision='full')
    train_data_size = len(train_dataset)
    print("训练集的长度为：{}".format(train_data_size))
    dataloader = DataLoader(train_dataset, batch_size=train_data_size, drop_last=True)

    # 鉴别器数据尺寸尝试
    # Dis = Discriminator()
    # for imgs,target in dataloader:
    #     # x = Dis(imgs)
    #     print(imgs.shape)
    #     alpha = torch.FloatTensor(np.random.random((9, 1, 1, 1)))
    #     print(alpha)
    #     interpolates = (alpha * imgs ).requires_grad_(True)
    #     print(interpolates.shape)

    # 生成器数据尺寸尝试
    # Gen = Generator()
    # z = Variable(torch.FloatTensor(np.random.normal(0, 1, (9, opt.latent_dim))))
    # gen_img = Gen(z)
    # print(gen_img.shape)
    # for img,gt in dataset:
    #     img = img[0][:][3][3]
    #     print(img.shape)
    #     print(gt.shape)

    # Loss functions Adam的损失函数
    # adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # GPU加速
    if cuda:
        generator.cuda()
        discriminator.cuda()
        # adversarial_loss.cuda()
    # --------------------------------
    # Optimizers adam优化器，带动量，理论不适用于wgan
    # ---------------------------------
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # --------------------------------
    # Optimizers 不带动量的优化器，wgan的理论
    # ---------------------------------
    # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr_gen)
    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr_dis)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    # ----------
    #  Training
    # ----------

    # tensorboard 的加载使用
    if opt.tensorboard:
        writer = SummaryWriter("logs")
    total_train_step = 0

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

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs)
            g_loss = -torch.mean(validity)
            if cuda:
                g_loss.cuda()

            # g_loss = adversarial_loss(validity, valid)

            g_loss.backward(retain_graph=True)
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach())

            # --------------------------------------------------------------
            # WGAN-GP鉴别器损失函数表达
            # --------------------------------------------------------------
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + 10 * gradient_penalty

            # --------------------------------------------------------------
            # WGAN鉴别器损失函数表达
            # --------------------------------------------------------------
            # d_loss = -torch.mean(validity_real) + torch.mean(validity_fake)
            if cuda:
                d_loss.cuda()

            # Clip weights of discriminator
            # for p in discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            # --------------------------------------------------------------
            # 为防止d-loss趋近于0，当d-loss太小时，不对鉴别器进行梯度更新，不进行训练，优化结果
            # --------------------------------------------------------------
            # if d_loss > 0:
            #     d_loss.backward(retain_graph=True,)
            #     optimizer_D.step()

            d_loss.backward()
            optimizer_D.step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )

            batches_done = epoch * len(dataloader) + i
            total_train_step = total_train_step + 1

            print("第 %d 轮训练已经完成！！！" % total_train_step)

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
        torch.save(discriminator.state_dict(), "dis_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                     opt.training_sample))
