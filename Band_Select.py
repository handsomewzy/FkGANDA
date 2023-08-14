import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import datasets
import utils
import model_data
import wgan3D

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
# 训练集和测试集划分,和分类网络保持一致
train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')
print("训练集大小: {} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
train_dataset = datasets.HyperX_choose(img, train_gt, patch_size=opt.img_size, choose_labels=opt.choose_labels,
                                       center_pixel=True, supervision='full')
train_data_size = len(train_dataset)
print("训练集的长度为：{}".format(train_data_size))
dataloader = DataLoader(train_dataset, batch_size=train_data_size, drop_last=True)



generator = model_data.Generator()
generator.load_state_dict(torch.load("gan_7_PaviaC_0.000800.pth"))

num = 1
for image, target in train_dataset:
    print('第 %d 样本的结果为-------------------------' % (num))
    # 将原始图片切片
    p1, p2, p3, p4, p5 = torch.chunk(input=image.type(torch.FloatTensor), chunks=5, dim=1)

    if num == 1:
        print("原始数据集的样本大小为 {}".format(image.shape))
        print("划分的五个片段的样本大小依次为：{}，{}，{}，{}，{}".format(
            p1.shape, p2.shape, p3.shape, p4.shape, p5.shape
        ))
    num += 1

    # 生成图片并将生成的图片切片
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (9, opt.latent_dim))))
    gen_img = generator(z)[0].type(torch.FloatTensor)
    # gen_img = gen_img.detach().numpy()
    f1, f2, f3, f4, f5 = torch.chunk(input=gen_img, chunks=5, dim=1)
    f1, f2, f3, f4, f5 = f1.detach(), f2.detach(), f3.detach(), f4.detach(), f5.detach()

    if num == 2:
        print("生成的数据的样本大小为 {}".format(gen_img.shape))
        print("划分的五个片段的样本大小依次为：{}，{}，{}，{}，{}".format(
            f1.shape, f2.shape, f3.shape, f4.shape, f5.shape
        ))

    # cov1 = np.corrcoef(np.vstack((f1,p1)))[0][1]
    # print(cov1)
    # std_list1 = map(torch.std(),p_list)
    # print(std_list1)

    # std1 = torch.std(p1)
    # std2 = torch.std(p2)
    # std3 = torch.std(p3)
    # std4 = torch.std(p4)
    # std5 = torch.std(p5)

    # 信息散度排序选择波段
    # m1 = torch.mean(p1)
    # m2 = torch.mean(p2)
    # m3 = torch.mean(p3)
    # m4 = torch.mean(p4)
    # m5 = torch.mean(p5)

    # 生成对应的五个高斯分布，为后续计算kl散度
    # g1 = np.random.normal(loc=m1, scale=std1, size=(1, 40, 7, 7))
    # g1 = torch.from_numpy(g1)
    #
    # g2 = np.random.normal(loc=m2, scale=std2, size=(1, 40, 7, 7))
    # g2 = torch.from_numpy(g2)
    #
    # g3 = np.random.normal(loc=m3, scale=std3, size=(1, 40, 7, 7))
    # g3 = torch.from_numpy(g3)
    #
    # g4 = np.random.normal(loc=m4, scale=std4, size=(1, 40, 7, 7))
    # g4 = torch.from_numpy(g4)
    #
    # g5 = np.random.normal(loc=m5, scale=std5, size=(1, 40, 7, 7))
    # g5 = torch.from_numpy(g5)

    # 计算生成图片切片后的片段和对应的真实片段的KL散度，衡量生成质量的好坏
    kl1 = F.kl_div(input=f1.softmax(dim=-1).log(), target=p1.softmax(dim=-1), reduction='sum')
    kl2 = F.kl_div(input=f2.softmax(dim=-1).log(), target=p2.softmax(dim=-1), reduction='sum')
    kl3 = F.kl_div(input=f3.softmax(dim=-1).log(), target=p3.softmax(dim=-1), reduction='sum')
    kl4 = F.kl_div(input=f4.softmax(dim=-1).log(), target=p4.softmax(dim=-1), reduction='sum')
    kl5 = F.kl_div(input=f5.softmax(dim=-1).log(), target=p5.softmax(dim=-1), reduction='sum')
    kl_list = [kl1, kl2, kl3, kl4, kl5]
    # print(kl1,kl2,kl3,kl4,kl5)

    f1, f2, f3, f4, f5 = f1.reshape(1, -1), f2.reshape(1, -1), \
                         f3.reshape(1, -1), f4.reshape(1, -1), f5.reshape(1, -1)
    f_list = [f1, f2, f3, f4, f5]

    l1 = p1.reshape(1, -1)
    l2 = p2.reshape(1, -1)
    l3 = p3.reshape(1, -1)
    l4 = p4.reshape(1, -1)
    l5 = p5.reshape(1, -1)
    l_list = [l1, l2, l3, l4, l5]

    # l6 = p6.reshape(1, -1)
    # l7 = p7.reshape(1, -1)
    # l8 = p8.reshape(1, -1)
    # l9 = p9.reshape(1, -1)
    # l10 = p10.reshape(1, -1)

    # 五个片段的相关系数矩阵，i j位置是片段j和片段j的相关系数
    # x = np.vstack((l1, l2, l3, l4, l5))
    # b = np.corrcoef(x)

    # print(b)
    # print(b[0][1])
    # print(np.linalg.det(b))

    # 真实片段分片后的方差
    std1 = torch.std(p1)
    std2 = torch.std(p2)
    std3 = torch.std(p3)
    std4 = torch.std(p4)
    std5 = torch.std(p5)
    std_list = [std1, std2, std3, std4, std5]
    # print(std_list)

    # std6 = torch.std(p6)
    # std7 = torch.std(p7)
    # std8 = torch.std(p8)
    # std9 = torch.std(p9)
    # std10 = torch.std(p10)
    # std_list = [std1, std2, std3, std4, std5,std6,std7,std8,std9,std10]
    # print('片段1到片段5的方差为--------')
    # print(std1,std2,std3,std4,std5)
    # print('片段6到片段10的方差为--------')
    # print(std6,std7,std8,std9,std10)

    # 选择3个
    max = -100
    for i in range(5):
        for j in range(i + 1, 5):
            for k in range(j + 1, 5):
                # 三波段的OIF指标计算
                # oif = (std_list[i] + std_list[j] + std_list[k]) / (math.fabs(x[i][j]) + math.fabs(x[i][k]) \
                #                                                    + math.fabs(x[j][k]))
                # cov_p = (math.fabs(x[i][j]) + math.fabs(x[i][k]) \
                #          + math.fabs(x[j][k]))

                # 三波段的信息熵和,kl散度越小越好，认为样本是趋近于高斯分布的。也是取负
                kl = (kl_list[i] + kl_list[j] + kl_list[k]) / (
                            kl_list[0] + kl_list[1] + kl_list[2] + kl_list[3] + kl_list[4])

                # 三波段的生成质量与原始图片的相关系数，越差的部分不应该保留，应该由原始数据替换，好的部分留下，因此取负
                cov_fp = (np.corrcoef(np.vstack((f_list[i], l_list[i])))[0][1] \
                          + np.corrcoef(np.vstack((f_list[j], l_list[j])))[0][1] \
                          + np.corrcoef(np.vstack((f_list[k], l_list[k])))[0][1]) / (
                                     np.corrcoef(np.vstack((f_list[0], l_list[0])))[0][1] \
                                     + np.corrcoef(np.vstack((f_list[1], l_list[1])))[0][1] \
                                     + np.corrcoef(np.vstack((f_list[2], l_list[2])))[0][1] +
                                     np.corrcoef(np.vstack((f_list[3], l_list[3])))[0][1] \
                                     + np.corrcoef(np.vstack((f_list[4], l_list[4])))[0][1])

                # 方差指标，越大信息量越大
                std = (std_list[i] + std_list[j] + std_list[k]) / (std_list[0] + std_list[1] \
                                                                   + std_list[2] + std_list[3] + std_list[4])
                # oif ,kl = oif.numpy(),kl.numpy()
                # print(oif,kl,cov_fp,std)

                score = 1 * kl + 1 * std

                # score = -cov_fp-kl+std+oif
                # print(oif)
                # print(kl)
                # print(cov_fp)
                if score > max:
                    max = score
                    i_ = i
                    j_ = j
                    k_ = k
    else:
        # print(max)
        # percent = (std_list[i]+std_list[j]+std_list[k]) / (std_list[0]+std_list[1]\
        #           +std_list[2]+std_list[3]+std_list[4])
        # print('所选波段占据全部波段的信息量比例为--------------')
        # print(percent)
        print('得分最高的的三波段组合为: %d ,%d, %d ' % (i_, j_, k_))
        # print(i_,j_,k_)

    # 选择5个
    # for i in range(10):
    #     for j in range(i+1,10):
    #         for k in range(j+1,10):
    #             for l in range(k+1,10):
    #                 for m in range(l+1,10):
    #                     oif = (std_list[i]+std_list[j]+std_list[k])**2 / (x[i][j])**2+(x[i][k])**2+(x[j][k])**2
    #                     if oif > max:
    #                         max = oif
    #                         print('OIF最好的5波段组合为----------------')
    #                         print(i,j,k,l,m)
