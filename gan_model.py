import argparse
import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
import datasets


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.0001, help="adam: learning rate of the generator")
parser.add_argument("--lr_dis", type=float, default=0.0001, help="adam: learning rate of the discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=17, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=3, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=200, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--dilation", type=int, default=1, help="conv3d hyperparameters")
parser.add_argument("--training_sample", type=float, default=0.2,
                    help="Percentage of samples to use for training (default: 10%)")
parser.add_argument("--choose_labels", default=[1, 7, 9, 16], help="the target we choose to generate")

opt = parser.parse_args()
print(opt)

img_shape = (1, opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


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
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1, out_channels=10, kernel_size=(2, 2, 2), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=10, out_channels=20, kernel_size=(2, 2, 2), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3)),
            nn.ConvTranspose3d(in_channels=20, out_channels=10, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                               padding=(1, 0, 0), dilation=(1, 1, 1)),
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
        self.l1 = nn.Linear(277, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 16)
        self.l4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.Conv3d(in_channels=20, out_channels=35, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.Conv3d(in_channels=35, out_channels=35, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.Conv3d(in_channels=35, out_channels=20, kernel_size=(3, 1, 1), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.Conv3d(in_channels=20, out_channels=10, kernel_size=(2, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.Flatten(),
        )
    def forward(self, imgs, target):
        x = self.model(imgs)
        x = torch.cat((x, self.label_embedding(target)), -1)
        x = F.leaky_relu(self.l1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        return x


# 模型的加载
Gen_model = Generator()
Gen_model.load_state_dict(torch.load("gan_model_10000.pth"))
# print(Gen_model)
Dis_model = Discriminator()
Dis_model.load_state_dict(torch.load("dis_model_10000.pth"))
# print(Dis_model)

if cuda:
    Gen_model.cuda()
    Dis_model.cuda()



z = Variable(FloatTensor(np.random.normal(0, 1, (16, opt.latent_dim))))
# print(z.shape)
gen_labels = Variable(LongTensor(np.random.choice(opt.choose_labels, 16)))
# print(gen_labels.shape)
gen_imgs = Gen_model(z, gen_labels)
print(len(gen_imgs))

# img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name='IndianPines',
#                                                                               target_folder="./dataset")
#
#
# dataset = datasets.HyperX(img, gt, patch_size=opt.img_size, ignored_labels=ignored_value, flip_augmentation=False,
#                           radiation_augmentation=False, mixture_augmentation=False, center_pixel=True,
#                           supervision='full')
#
# dataloader = DataLoader(dataset, batch_size=16, drop_last=True)

class Create_dataset(torch.utils.data.Dataset):
    def __init__(self, gen_imgs, gen_labels):
        super(Create_dataset, self).__init__()
        self.data = gen_imgs
        self.gt = gen_labels

    def __getitem__(self, i):
        data = self.data[i]
        label = self.gt[i]
        return data, label

    def  __len__(self):
        return len(gen_imgs)

MyDataset = Create_dataset(gen_imgs=gen_imgs,gen_labels=gen_labels)
# New_dataset = MyDataset + dataset
# print(len(dataset))
# print(len(New_dataset))
# # for i,j in New_dataset:
# #     print(i.shape)
# #     print(j)





