import argparse
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import KeepGAN
import visdom
import seaborn as sns
import models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import datasets
import utils
import model_data
import wgan3D

from mini_classify import MiniClassify
from models import test


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


seed_torch()
# 和生成器参数保持一致
opt = wgan3D.opt
img_shape = wgan3D.img_shape
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 数据集的加载
img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                              target_folder="Datasets")
# 训练集和测试集划分,和分类网络保持一致,划分验证集辅助
total_train_gt, test_gt = utils.sample_gt(gt, 0.1, mode='random')
train_gt, val_gt = utils.sample_gt(total_train_gt, 0.9, mode='random')


def create_mixed_data(p_list, g_list, mask):
    """

    :param p_list: 真实的图片切片元组
    :param g_list: 生成的图片切片元组和真实的有相同的大小
    :param mask: 用来选择的面具函数，依靠评分函数得到
    :return: 返回一个全新的按照mask交叉的新数据
    """
    new_data = []
    for index, i in enumerate(mask):
        if i == 1:
            new_data.append(p_list[index])
        else:
            new_data.append(g_list[index])
    new = torch.cat(new_data, 2)
    return new


def compute_mask(p_list, g_list, percent=0.5):
    """

    :param percent: 选择交叉的比例，默认为一半0.5
    :param p_list: 真实的图片切片元组
    :param g_list: 生成的图片切片元组和真实的有相同的大小
    :return: 返回一个和真实图片元组一样大小的mask列表，用于后续生成新的交叉样本
    """
    # 计算对应划分波段的KL散度
    kl_list = []
    for i in range(len(p_list)):
        kl_list.append(
            F.kl_div(input=p_list[i].softmax(dim=-1).log(), target=g_list[i].softmax(dim=-1), reduction='sum'))

    # 计算划分的各个波段的方差
    std_list = []
    for i in p_list:
        std_list.append(torch.std(i))

    # KL散度越大，表明生成质量越差，不应该保留，应该使用原始波段，取正
    # STD方差越大，表明信息量越大，应该使用原始波段，取正
    # score = KL + STD

    mask = np.zeros_like(p_list)  # 用来选择使用真实数据还是生成的数据
    count = 0  # 记录循环的次数
    score = 0  # 记录分数
    max = 0  # 记录最大的分数
    index_max = 0
    while count < (percent * len(p_list)):
        for index, i in enumerate(mask):
            if i == 0:
                # 每次都选择得分最高的波段
                score = std_list[index] + kl_list[index]
                if score > max:
                    max = score
                    index_max = index
        else:
            mask[index_max] = 1
            count = count + 1
            max = 0  # 初始化最大分数
            index_max = 0
    # print(mask)
    return mask


def create_new_dataset(pth="gan_2_PaviaC_0.000800.pth", label=2, copy=False, gan=False, num=5, percent=0.6):
    """

    :param pth: 预先训练好的GAN生成器模型参数地址
    :param label: 想要生成的数据标签
    :param copy: bool数据，用来控制是否进行纯copy扩增，默认为False
    :param gan: bool数据，用来控制是否进行纯gan扩增，默认为False
    :return: 返回一个和训练集相同大小的交叉的新数据集，和train_gt有相同的大小
    """
    seed_torch()
    dataset = datasets.HyperX_choose(img, total_train_gt, choose_labels=[label],
                                     dataset=setting.dataset_name, patch_size=7,
                                     flip_augmentation=False,
                                     radiation_augmentation=False,
                                     mixture_augmentation=False, ignored_labels=[0],
                                     center_pixel=True, supervision='full')
    dataloader = DataLoader(dataset, batch_size=num, drop_last=True, pin_memory=False)

    # keepGAN的加载，后续函数改名字
    # generator = model_data.Generator()

    # WGAN的加载，后续读取的pth记得改成对应名字
    generator = model_data.WGAN_Generator()

    # DCGAN的加载，后续读取的pth记得改成对应名字
    # generator = model_data.DCGAN_Generator(3)

    # HSGAN的加载，后续读取的pth记得改成对应名字
    # generator = model_data.HSGAN_Generator(num)

    # 3DGAMO的加载，后续读取的pth记得改成对应名字
    # generator = model_data._3DGAMO(bands=opt.channels)

    # AC-WGAN-GP的加载，后续读取的pth记得改成对应名字
    # generator = model_data.ACWGANGP()

    if torch.cuda.is_available():
        generator.cuda()
    generator.load_state_dict(torch.load(pth, map_location='cuda:0'))

    gen_labels = Variable(torch.LongTensor(np.random.choice([label], num)))

    # 生成数据并切分，和原始数据切分成一样的大小,WGAN, keepGAN, 3DGAMO, 输入（batch size，100）
    z = Variable(FloatTensor(np.random.normal(0, 1, (num, opt.latent_dim))))

    # DCGAN输入（batch size，100，1,1）
    # z = Variable(FloatTensor(np.random.normal(0, 1, (num, opt.latent_dim, 1, 1))))

    # HSGAN输入（batch size，1,100）
    # z = Variable(Tensor(np.random.normal(0, 1, (num, 1, opt.latent_dim))))

    # ACWGANGP的输入，需要有z和labels，同时有两个。同时生成的图片也需要传入两个
    # z = Variable(FloatTensor(np.random.normal(0, 1, (num, opt.latent_dim))))
    # gen_label = Variable(LongTensor(np.random.randint(label, label + 1, num)))
    # gen_img = generator(z, gen_label).type(torch.FloatTensor)
    # print(gen_imgs.shape)

    # 其余GAN使用下面的生成器，只需要传入噪声z
    gen_img = generator(z).type(torch.FloatTensor)
    g_list = torch.chunk(input=gen_img, chunks=opt.channels, dim=2)
    out_flag = 0
    for image, target in dataloader:
        if out_flag == 0:
            # 切分原始数据,生成对应通道数的元组数据,每一个波段单独切分
            p_list = torch.chunk(input=image.type(torch.FloatTensor), chunks=opt.channels, dim=2)

            # 加入copy和纯GAN扩增策略
            if copy:
                mask = np.ones_like(p_list)
            elif gan:
                mask = np.zeros_like(p_list)
            else:
                mask = compute_mask(p_list, g_list, percent=percent)
            print("数据种类{}的交叉比例如下！！！".format(label))
            print(mask)
            out_flag = out_flag + 1
        else:
            break

        # 因为波段选择的时候要考虑生成波段的质量，因此先使用生成波段选择出特征波段，之后进行0-1的正态分布
        # g_list_normal = Variable(torch.FloatTensor(np.random.normal(0, 1, (num, 1, 200, 7, 7))))
        # 空波段，全是0
        g_list_normal = Variable(torch.FloatTensor(np.zeros((num, 1, 200, 7, 7))))
        g_list = torch.chunk(input=g_list_normal, chunks=opt.channels, dim=2)
        new = create_mixed_data(p_list, g_list, mask)

        # 正常的进行交叉生成
        # new = create_mixed_data(p_list, g_list, mask)
    return datasets.Create_dataset(gen_imgs=new, gen_labels=gen_labels)


def show_results(results, label_values=None):
    """

    :param results: 之前计算出来的一个字典集合，包含了数据的混淆矩阵、准确度、F1分数、Kappa分数等
    :param label_values: 所选数据集的标签名称
    """
    text = ""
    cm = results["Confusion matrix"]
    accuracy = results["Accuracy"]
    F1scores = results["F1 scores"]
    kappa = results["Kappa"]
    accuracies_every = results["Accuracy_scores"]
    AA = results["AA"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    text += "Overall Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"
    text += "Average Accuracy : {:.03f}%\n".format(AA)
    text += "---\n"

    text += "F1 scores :\n"
    for label, score in zip(label_values, F1scores):
        text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    text += "Kappa: {:.03f}\n".format(kappa)

    # 输出每一类的精度
    text += "每一类的精确度如下："
    for label, score in zip(label_values, accuracies_every):
        text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    print(text)


def create_total_dataset(origin_dataset=None, add_labels=None, num_list=None, mix_percent=0.6):
    """

    :param origin_dataset: 选择扩增的基础训练集，原始的训练数据集
    :param num_list: 每一个样本选择扩增的个数
    :param mix_percent: 每一个样本选择的扩增比例，在这里是统一的，后续应该也是类似的一个列表，每一个数据是不一样的。
    :return: 返回一个加入扩增数据后的全新的数据集。直接可以用于后续的训练分类任务。
    """
    mix_percent = mix_percent
    the_create_dataset = origin_dataset
    num_list = num_list
    # 针对KSC数据集的循环
    # for i in range(12):
    #     the_create_dataset += create_new_dataset(pth="gan_{}_KSC_0.050000.pth".format(i + 1)
    #                                              , label=i + 1, num=num_list[i], percent=mix_percent)
    # 针对Botswana数据集的循环
    # for i in range(13):
    #     the_create_dataset += create_new_dataset(pth="gan_{}_Botswana_0.050000.pth".format(i + 1)
    #                                              , label=i + 1, num=num_list[i], percent=mix_percent)
    # [2, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16]
    # [1,3,5,7,10,12,15]
    j = 0
    for i in add_labels:
        the_create_dataset += create_new_dataset(pth="WGANGP_{}_IndianPines_0.100000.pth".format(i)
                                                 , label=i, num=num_list[j], percent=mix_percent)
        j = j + 1
    gc.collect()
    torch.cuda.empty_cache()
    return the_create_dataset


if __name__ == "__main__":
    seed_torch()
    parser = argparse.ArgumentParser()
    # 针对数据集的一些参数设置
    parser.add_argument("--channels", type=int, default=200, help="number of image channels"
                                                                  "IndianPines 200 17"
                                                                  "PaviaC 102 10"
                                                                  "PaviaU 103 10"
                                                                  "KSC 176 15"
                                                                  "Botswana 145 16"
                                                                  "Houston 144 16"
                                                                  "salinas 204 17")
    parser.add_argument("--classes", type=int, default=17)
    parser.add_argument("--patch_size", type=int, default=7)

    parser.add_argument('--dataset_name', type=str, default='IndianPines', help="使用扩增的数据集名称,如下："
                                                                                "PaviaC"
                                                                                "IndianPines")
    parser.add_argument("--training_sample", type=float, default=0.1,
                        help="Percentage of samples to use for training (default: 10%)"
                             "应该和扩增网络的训练集的参数保持一致")
    # 针对不同扩增方法的一些参数设置
    parser.add_argument('--flip_augmentation', action='store_true', default=False,
                        help="Random flips (if patch_size > 1)")
    parser.add_argument('--radiation_augmentation', action='store_true', default=False,
                        help="Random radiation noise (illumination)")
    parser.add_argument('--mixture_augmentation', action='store_true', default=False,
                        help="Random mixes between spectra")
    parser.add_argument('--KeepGAN', default=True, help="是否选择利用gan进行小数据增强")
    parser.add_argument('--total_KeepGAN', default=False, help="是否将初始样本替换为keepGAN生成的数据")
    setting = parser.parse_args()

    # visdom 可视化部分
    viz = visdom.Visdom(env=setting.dataset_name)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    # 数据集的加载
    img, gt, label_value, ignored_value, rgb_bands, palette = datasets.get_dataset(dataset_name=setting.dataset_name,
                                                                                   target_folder="Datasets")
    # print(img.keys())
    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(label_value) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}


    def convert_to_color(x):
        return utils.convert_to_color_(x, palette=palette)


    def convert_from_color(x):
        return utils.convert_from_color_(x, palette=invert_palette)


    mini_train_gt, mini_test_gt = utils.sample_gt(gt, setting.training_sample, mode='random')
    # 训练集和测试集划分,和分类网络保持一致,划分验证集辅助
    # 剩下两个是后续训练集和验证集的划分。
    total_train_gt, test_gt = utils.sample_gt(gt, setting.training_sample, mode='random')

    # 人为制造出小样本数据
    # KSC数据集的制造选择
    # mini_labels = [6, 7, 8, 13]
    # normal_labels = [1, 2, 3, 4, 5, 9, 10, 11, 12, 14, 15]
    # mini_train_dataset = datasets.HyperX_choose(img, mini_train_gt, choose_labels=mini_labels,
    #                                             dataset=setting.dataset_name, patch_size=setting.patch_size,
    #                                             flip_augmentation=False,
    #                                             radiation_augmentation=False,
    #                                             mixture_augmentation=False, ignored_labels=[0],
    #                                             center_pixel=True, supervision='full')
    # normal_train_dataset = datasets.HyperX_choose(img, total_train_gt, choose_labels=normal_labels,
    #                                               dataset=setting.dataset_name, patch_size=setting.patch_size,
    #                                               flip_augmentation=False,
    #                                               radiation_augmentation=False,
    #                                               mixture_augmentation=False, ignored_labels=[0],
    #                                               center_pixel=True, supervision='full')
    # total_train_dataset = mini_train_dataset + normal_train_dataset

    # 统一一样的选择比例
    total_train_dataset = datasets.HyperX(img, total_train_gt, dataset=setting.dataset_name, patch_size=7,
                                          flip_augmentation=False,
                                          radiation_augmentation=False,
                                          mixture_augmentation=False, ignored_labels=[0],
                                          center_pixel=True, supervision='full')

    total_train_datasize = len(total_train_dataset)
    print("训练集的长度为：{}".format(total_train_datasize))

    # add_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    add_labels = [1, 3, 7, 9, 15]
    if setting.flip_augmentation:
        add_train_dataset = datasets.HyperX_choose(img, mini_train_gt, choose_labels=add_labels,
                                                   dataset=setting.dataset_name, patch_size=setting.patch_size,
                                                   flip_augmentation=True,
                                                   radiation_augmentation=False,
                                                   mixture_augmentation=False, ignored_labels=[0],
                                                   center_pixel=True, supervision='full')
        print("生成的训练集的长度为：{}".format(len(add_train_dataset)))
        new_train_dataset = total_train_dataset + add_train_dataset
        print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))

    elif setting.radiation_augmentation:
        add_train_dataset = datasets.HyperX_choose(img, mini_train_gt, choose_labels=add_labels,
                                                   dataset=setting.dataset_name, patch_size=setting.patch_size,
                                                   flip_augmentation=False,
                                                   radiation_augmentation=True,
                                                   mixture_augmentation=False, ignored_labels=[0],
                                                   center_pixel=True, supervision='full')
        print("生成的训练集的长度为：{}".format(len(add_train_dataset)))
        new_train_dataset = total_train_dataset + add_train_dataset
        print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))

    elif setting.mixture_augmentation:
        add_train_dataset = datasets.HyperX_choose(img, mini_train_gt, choose_labels=add_labels,
                                                   dataset=setting.dataset_name, patch_size=setting.patch_size,
                                                   flip_augmentation=False,
                                                   radiation_augmentation=False,
                                                   mixture_augmentation=True,
                                                   center_pixel=True, supervision='full')
        print("生成的训练集的长度为：{}".format(len(add_train_dataset)))
        new_train_dataset = total_train_dataset + add_train_dataset
        print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))

    elif setting.KeepGAN:
        # KSC
        # num_list = [38, 12, 13, 13, 8, 11, 5, 21, 26, 20, 21, 25, 46]
        # num_list = [13,13,8,11,21]

        # Botswana
        # num_list = [14, 5, 13, 11, 13, 13, 13, 10, 16, 12, 15, 9, 13, 5]
        # num_list = [5, 13, 13, 5]

        # IP 1 3 4 5 7 9 10 12 13 15
        # num_list = [5, 143, 74, 23, 41, 73, 3, 48, 2, 97, 234, 58, 20, 126, 34]
        num_list = [5, 72, 3, 2, 30]

        # pu数据集
        # num_list = [32, 86, 10, 15, 7, 25, 7, 18, 5]
        # num_list = [9,7,  25, 7]

        # pc数据集 0.002
        # num_list = [130, 15, 6, 5, 13, 18, 15, 86, 6]
        # num_list = [6, 12, 18, 15]

        # houston数据集 0.01
        # num_list = [14, 14, 8, 13, 13, 3, 15, 14, 15, 14, 16, 14, 6, 5, 8]
        # num_list = [3, 15, 14, 6]

        # salinas数据集 0.01
        # num_list = [20, 37, 19, 13, 56, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]
        # num_list = [110,32,70,16]

        # 我们提出的keepGAN中，交叉比例设置为0.6，其余GAN的时候，全部用生成的数据，交叉比例设置为0
        new_train_dataset = create_total_dataset(origin_dataset=total_train_dataset, add_labels=add_labels,
                                                 num_list=num_list, mix_percent=0)

        # new_dataset_6 = create_new_dataset(pth="gan_{}_PaviaU_0.005000.pth".format(6)
        #                                    , label=6, num=25, percent=0.6)
        # new_train_dataset = new_train_dataset + new_dataset_6

        print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))

    elif setting.total_KeepGAN:
        # KSC数据集
        # num_list = [12, 13, 13, 8, 11, 5, 21, 26, 20, 21, 25, 46]

        # bw数据集
        # num_list = [5, 13, 11, 13, 13, 13, 10, 16, 12, 15, 9, 13, 5]

        # pu数据集
        num_list = [10, 7, 25]
        new_dataset = create_new_dataset(pth="gan_{}_PaviaU_0.005000.pth".format(1)
                                         , label=1, num=32, percent=0.6)
        new_train_dataset = create_total_dataset(origin_dataset=new_dataset,
                                                 add_labels=[3, 5, 6],
                                                 num_list=num_list, mix_percent=0.6)

        print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))





    else:
        new_train_dataset = total_train_dataset

    train_dataloader = DataLoader(new_train_dataset,
                                  batch_size=100,
                                  shuffle=True)

    # 不同分类器的选择,均是一些神经网络
    classify = models.Baseline(input_channels=setting.channels, n_classes=setting.classes)
    # classify = models.LeeEtAl(in_channels=setting.channels, n_classes=setting.classes)
    # classify = models.HuEtAl(input_channels=setting.channels, n_classes=setting.classes)
    # classify = models.HamidaEtAl(input_channels=setting.channels, n_classes=setting.classes,
    #                              patch_size=setting.patch_size, dilation=1)
    # classify = models.LiEtAl(setting.channels, setting.classes, n_planes=16, patch_size=setting.patch_size)
    # classify = models.HeEtAl(setting.channels, setting.classes, patch_size=setting.patch_size)

    if cuda:
        classify.cuda()

    # 对于不同网络，不同的优化器选择
    # FNN
    optimizer = torch.optim.Adam(classify.parameters(), lr=0.0005)
    # LeeEtAl
    # optimizer = torch.optim.Adam(classify.parameters(), lr=0.001)
    # HuEyAl
    # optimizer = torch.optim.SGD(classify.parameters(), lr=0.01)
    # hamida
    # optimizer = torch.optim.SGD(classify.parameters(), lr=0.005, weight_decay=0.0005)
    # li
    # optimizer = torch.optim.SGD(classify.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    # he
    # optimizer = torch.optim.Adagrad(classify.parameters(), lr=0.01, weight_decay=0.01)
    # 确定损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # 训练过程
    for i in range(20000):
        for mixed_imgs, targets in train_dataloader:
            if cuda:
                mixed_imgs = mixed_imgs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()

            # _, t, c, w, h = mixed_imgs.size()
            # print(mixed_imgs.shape)
            # mixed_imgs = mixed_imgs.reshape(_, t * c * w * h)
            # print(mixed_imgs.shape)

            outputs = classify(mixed_imgs)
            # print(outputs.shape)
            # print(targets.shape)
            loss = loss_fn(outputs, targets)

            if cuda:
                loss.cuda()

            loss.backward(retain_graph=True)
            optimizer.step()

        # 每迭代一定轮次，在验证集测试一下结果
        if i % 500 == 0:
            hyperparams = {'patch_size': setting.patch_size, 'center_pixel': True, 'batch_size': 100,
                           'device': torch.device('cuda:{}'.format(0)), 'n_classes': setting.classes,
                           'test_stride': 1}
            probabilities = test(classify, img, hyperparams)
            prediction = np.argmax(probabilities, axis=-1)
            run_results = utils.metrics(prediction, test_gt, ignored_labels=[0], n_classes=setting.classes)

            show_results(run_results, label_values=label_value)

            # visdom 可视化部分
            mask = np.zeros(gt.shape, dtype='bool')
            for l in [0]:
                mask[gt == l] = True
            prediction[mask] = 0

            color_prediction = convert_to_color(prediction)
            # utils.display_predictions(color_prediction, viz, gt=None,
            #                           caption="epoch = {} ,Prediction vs. test ground truth".format(i))
            utils.display_predictions(color_prediction, viz, gt=convert_to_color(test_gt),
                                      caption="epoch = {} ,Prediction vs. test ground truth".format(i))

        print('oh yeah 第 {} 轮训练完成,loss = {}'.format(i, loss))




    # 训练完成后，在测试集进行测试，得到最终结果。
    else:
        hyperparams = {'patch_size': setting.patch_size, 'center_pixel': True, 'batch_size': 100,
                       'device': torch.device('cuda:{}'.format(0)), 'n_classes': setting.classes,
                       'test_stride': 1}
        probabilities = test(classify, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        run_results = utils.metrics(prediction, test_gt, ignored_labels=[0], n_classes=setting.classes)

        show_results(run_results, label_values=label_value)
