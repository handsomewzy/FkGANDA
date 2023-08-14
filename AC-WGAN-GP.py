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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.adv_layer = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 17),
            nn.Softmax()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.adv_layer(img_flat)
        label = self.aux_layer(img_flat)
        return validity, label


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # 此处使用的是ACGAN，会输出两个结果，m1仅用来储存没用的信息
    d_interpolates, m1 = D(interpolates)
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


if __name__ == "__main__":
    for i in [8,10,15,16]:
        label = i
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

        # Loss weight for gradient penalty
        lambda_gp = 10

        # Loss functions
        adversarial_loss = torch.nn.BCELoss()
        auxiliary_loss = torch.nn.CrossEntropyLoss()

        # Initialize generator and discriminator
        generator = Generator()
        discriminator = Discriminator()

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
            auxiliary_loss.cuda()

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
            for i, (imgs, labels) in enumerate(dataloader):
                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                gen_labels = Variable(torch.LongTensor(np.random.randint(label, label+1, imgs.shape[0])))
                # print(gen_labels)

                if cuda:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    real_imgs = real_imgs.cuda()
                    labels = labels.cuda()
                    gen_labels = gen_labels.cuda()
                    z = z.cuda()

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)
                # print(gen_imgs.shape)

                # 分别计算ACGAN的损失和WGANGP的损失，将他们的和作为新的损失函数
                validity, pred_label = discriminator(gen_imgs)
                real_validity, p2 = discriminator(real_imgs)
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)

                g_loss_WGAN = -torch.mean(validity)
                g_loss_ACGAN = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

                g_loss = g_loss_ACGAN + g_loss_WGAN

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # ACGAN的鉴别器损失
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
                d_loss = -d_real_loss + d_fake_loss + lambda_gp * gradient_penalty

                optimizer_D.step()

                batches_done = epoch * len(dataloader) + i
                total_train_step = total_train_step + 1

                print("第 %d 轮训练已经完成！！！d_loss = %f g_loss= %f g_loss_ACGAN= %f g_loss_WGAN= %f" % (
                    total_train_step, d_loss, g_loss, g_loss_ACGAN, g_loss_WGAN))

        else:
            print("总共 %d 轮次的训练已经完成！！！" % opt.n_epochs)
            # 模型的存储
            if opt.save_model:
                torch.save(generator.state_dict(), "ACWGANGP_%d_%s_%f.pth" % (opt.choose_labels[0], opt.dataset_name,
                                                                              opt.training_sample))
