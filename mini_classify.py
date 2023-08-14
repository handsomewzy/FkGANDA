import argparse
import os
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datasets
import utils


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


class MiniClassify(nn.Module):
    def __init__(self):
        super(MiniClassify, self).__init__()
        dilation = (1, 1, 1)
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=5),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=5, out_channels=5, kernel_size=(3, 1, 1), dilation=dilation, stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=5),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=(3, 3, 3), dilation=dilation, stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(num_features=1),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(2200, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 14),
            # nn.Softmax()
        )

    def forward(self, imgs):
        x = self.model(imgs)
        return x


# 使用生成的数据，训练一个小型的分类器，只输入生成的样本，然后使用和原始划分的验证集进行验证，
# 得到的结果作为分类分数，辅助我们进行交叉比例的选择。

# 加载分类器和优化器
# classify = MiniClassify()
# if cuda:
#     classify.cuda()
# optimizer = torch.optim.Adam(classify.parameters())
# dataloader = DataLoader(my_dataset2, batch_size=3, drop_last=True)
# test_dataset = datasets.HyperX_choose(img, test_gt, patch_size=opt.img_size, choose_labels=opt.choose_labels,
#                                       center_pixel=True, supervision='full')
# test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)
#
# # 训练过程
# for i in range(250):
#     for mixed_imgs, targets in dataloader:
#         if cuda:
#             mixed_imgs = Variable(mixed_imgs.type(FloatTensor))
#         optimizer.zero_grad()
#         loss_cls = torch.mean(classify(mixed_imgs))
#
#         if cuda:
#             loss_cls.cuda()
#         loss_cls.backward()
#         optimizer.step()
#     else:
#         score_cls = loss_cls
#         print("最终的生成图片的分类分数为： %f" % score_cls)
#
#     # 测试步骤
#     classify.eval()
#     total_test_loss = 0
#     with torch.no_grad():
#         for data in test_dataloader:
#             real_imgs, targets = data
#
#             if torch.cuda.is_available():
#                 real_imgs = real_imgs.cuda()
#
#             loss_real = torch.mean(classify(real_imgs))
#             final_score = total_test_loss + (loss_real - loss_cls)
#
#     print("最终的分数为： {}".format(final_score))

# PaviaU 数据集，除去样本5 9 都进行扩增
# my_dataset1 = create_new_dataset(pth="gan_1_PaviaU_0.005000.pth", label=1, copy=copy_signal, gan=gan_signal, num=15)
# my_dataset2 = create_new_dataset(pth="gan_2_PaviaU_0.005000.pth", label=2, copy=copy_signal, gan=gan_signal, num=40)
# my_dataset3 = create_new_dataset(pth="gan_3_PaviaU_0.005000.pth", label=3, copy=copy_signal, gan=gan_signal, num=5)
# my_dataset4 = create_new_dataset(pth="gan_4_PaviaU_0.005000.pth", label=4, copy=copy_signal, gan=gan_signal, num=8)
# my_dataset6 = create_new_dataset(pth="gan_6_PaviaU_0.005000.pth", label=6, copy=copy_signal, gan=gan_signal, num=13)
# my_dataset7 = create_new_dataset(pth="gan_7_PaviaU_0.005000.pth", label=7, copy=copy_signal, gan=gan_signal, num=4)
# my_dataset8 = create_new_dataset(pth="gan_8_PaviaU_0.005000.pth", label=8, copy=copy_signal, gan=gan_signal, num=9)

# PaviaU 数据集，除去样本5 9 都进行扩增
# my_dataset1 = create_new_dataset(pth="gan_1_PaviaU_0.005000.pth", label=1, copy=copy_signal, gan=gan_signal, num=15)
# my_dataset2 = create_new_dataset(pth="gan_2_PaviaU_0.005000.pth", label=2, copy=copy_signal, gan=gan_signal, num=40)
# my_dataset3 = create_new_dataset(pth="gan_3_PaviaU_0.005000.pth", label=3, copy=copy_signal, gan=gan_signal, num=5)
# my_dataset4 = create_new_dataset(pth="gan_4_PaviaU_0.005000.pth", label=4, copy=copy_signal, gan=gan_signal, num=8)
# my_dataset6 = create_new_dataset(pth="gan_6_PaviaU_0.005000.pth", label=6, copy=copy_signal, gan=gan_signal, num=13)
# my_dataset7 = create_new_dataset(pth="gan_7_PaviaU_0.005000.pth", label=7, copy=copy_signal, gan=gan_signal, num=4)
# my_dataset8 = create_new_dataset(pth="gan_8_PaviaU_0.005000.pth", label=8, copy=copy_signal, gan=gan_signal, num=9)

# 样本1 7 9的数据扩增 IP数据集 五波段
# MyDataset_1 = model_data.create_data(1, pth="gan_model1.pth", num=2, img=img, gt=train_gt, data_name='IndianPines')
# MyDataset_7 = model_data.create_data(7, pth="gan_model_7wgan.pth", num=2, img=img, gt=train_gt, data_name='IndianPines')
# MyDataset_9 = model_data.create_data(9, pth="gan_model9.pth", num=1, img=img, gt=train_gt, data_name='IndianPines')

# 样本2 3 4 5 7的数据扩增，PaviaC数据集
# 0.0008训练比例
# my_dataset2 = create_new_dataset(pth="gan_2_PaviaC_0.000800.pth", label=2, copy=copy_signal, gan=gan_signal, num=3)
# my_dataset3 = create_new_dataset(pth="gan_3_PaviaC_0.000800.pth", label=3, copy=copy_signal, gan=gan_signal, num=2)
# my_dataset4 = create_new_dataset(pth="gan_4_PaviaC_0.000800.pth", label=4, copy=copy_signal, gan=gan_signal, num=1)
# my_dataset5 = create_new_dataset(pth="gan_5_PaviaC_0.000800.pth", label=5, copy=copy_signal, gan=gan_signal, num=3)
# my_dataset7 = create_new_dataset(pth="gan_7_PaviaC_0.000800.pth", label=7, copy=copy_signal, gan=gan_signal, num=3)

# 0.008训练比例，PaviaC数据集，3 4 7样本数据扩增
# my_dataset3 = create_new_dataset(pth="gan_3_PaviaC_0.008000.pth", label=3, copy=copy_signal, gan=gan_signal, num=11)
# my_dataset4 = create_new_dataset(pth="gan_4_PaviaC_0.008000.pth", label=4, copy=copy_signal, gan=gan_signal, num=10)
# my_dataset7 = create_new_dataset(pth="gan_7_PaviaC_0.008000.pth", label=7, copy=copy_signal, gan=gan_signal, num=29)

    # max = 0
    # for i in range(11):
    #     print(i / 10)
    #
    #     for j, (imgs, labels) in enumerate(dataloader):
    #         # print(imgs)
    #         mix_score = torch.mean(discriminator(imgs))
    #         mean_score = torch.mean(imgs)
    #         std_score = torch.std(imgs)
    #
    #         score = abs(mix_score)
    #         if score > max:
    #             max = score
    #             mix_percent = i / 10
    #     else:
    #         print("最终交叉的分数为：%.10f" % max)
    #         print("最终的交叉比例为：%f" % mix_percent)