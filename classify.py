import random
from utils import open_file
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import datasets
import utils
import numpy as np
from scipy import io
from torch.autograd import Variable

# IP 数据集的读取与加载

# train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
#                                           download=True)
# test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
#                                          download=True)
#
# train_dataloader = DataLoader(train_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)

# img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name='IndianPines',
#                                                                               target_folder="./dataset")
# # 训练集和测试集划分
# train_gt, test_gt = utils.sample_gt(gt, 0.8, mode='random')
# print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
#                                                  np.count_nonzero(gt)))
#
# # 训练集和验证集划分
# train_gt, val_gt = utils.sample_gt(train_gt, 0.95, mode='random')
#
# # 训练dataloader
# train_dataset = datasets.HyperX_choose(img, train_gt, patch_size=5, choose_labels=[1,9,14], flip_augmentation=False,
#                           radiation_augmentation=False, mixture_augmentation=False, center_pixel=True,
#                           supervision='full')
# train_dataloader = DataLoader(train_dataset, batch_size=16, drop_last=True)
#
# # 测试dataloader
# test_dataset = datasets.HyperX(img, test_gt, patch_size=5, ignored_labels=ignored_value, flip_augmentation=False,
#                           radiation_augmentation=False, mixture_augmentation=False, center_pixel=True,
#                           supervision='full')
# test_dataloader = DataLoader(test_dataset, batch_size=16, drop_last=True)
#
# # 验证dataloader
# val_dataset = datasets.HyperX(img, val_gt, patch_size=4, ignored_labels=ignored_value, flip_augmentation=False,
#                           radiation_augmentation=False, mixture_augmentation=False, center_pixel=True,
#                           supervision='full')
# val_dataloader = DataLoader(val_dataset, batch_size=16, drop_last=True)



# 数据集的长度
# train_data_size = len(train_dataset)
# test_data_size = len(test_dataset)
# print("训练集的长度为：{}".format(train_data_size))
# print("测试集的长度为：{}".format(test_data_size))

gt1 = io.loadmat('C:\\Users\\Wang_Zhaoyang\\Desktop\\Code\\Datasets\\IndianPines\\indianpines_disjoint_dset.mat')

print(1)
print(13//2)
"""
# 搭建神经网络，简单的卷积神经网络
class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=17),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 创建网络模型
model = Classify()
if torch.cuda.is_available():
    model = model.cuda()

# 确定损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器的选择
optimizer = torch.optim.SGD(model.parameters(),lr=0.005)

# 一些参数的设置
total_train_step = 0
total_test_step = 0
epoch = 100


# tensorboard 的加载使用
writer = SummaryWriter("logs")

# 开始训练
for i in range(epoch):
    print("------第{}轮训练开始--------".format(i+1))

    # 训练步骤
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = torch.resize_as_(imgs, torch.ones(16, 3, 32, 32))

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数： {} loss： {}".format(total_train_step, loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = torch.resize_as_(imgs, torch.ones(16, 3, 32, 32))

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            acc = (outputs.argmax(1) == targets).sum()
            total_acc = total_acc + acc
    print("整体测试集的损失为： {}".format(total_test_loss))
    print("整体测试集的精确率为： {}".format(total_acc/test_data_size))
    writer.add_scalar("test_loss",total_test_loss.item(),total_test_step)
    writer.add_scalar("test_acc", total_acc/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(model,"model_{}.pth".format(i))
    # print("模型已保存")

writer.close()
"""


# 从训练集中提取小样本数据
class HyperX_choose(torch.utils.data.Dataset):
    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX_choose, self).__init__()
        self.data = data
        self.label = gt

        self.patch_size = hyperparams['patch_size']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        self.choose_labels = hyperparams['choose_labels']

        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.zeros_like(gt)
            for l in self.choose_labels:
                mask[gt == l] = 1

        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        # np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN

        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label
