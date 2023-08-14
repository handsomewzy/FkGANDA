import argparse
import os


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

parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
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
parser.add_argument("--training_sample", type=float, default=0.1,
                    help="Percentage of samples to use for training (default: 10%)"
                         "应该和扩增网络的训练集的参数保持一致")
parser.add_argument("--choose_labels", default=[1], help="想要扩增的样本，一次选择一个，一次训练一个样本类别的网络")

parser.add_argument('--save_model', type=str, default=True, help="是否选择保存训练的模型")
parser.add_argument('--tensorboard', type=str, default=False, help="是否使用tensorboard记录训练数据")

opt = parser.parse_args()

# --------------------------------
# main函数中主要设定
# --------------------------------
# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
# PaviaC IndianPines
parser.add_argument('--dataset', type=str, default='IndianPines',
                    help="Dataset to use."
                         "PaviaC"
                         "IndianPines"
                         "PaviaU"
                         "Botswana"
                    )
parser.add_argument('--model', type=str, default='Baseline',
                    help="Model to train. Available:\n"
                         "SVM (linear), "
                         "SVM_grid (grid search on linear, poly and RBF kernels), "
                         "Baseline (fully connected NN), "
                         "hu (1D CNN), "
                         "hamida (3D CNN + 1D classifier), "
                         "lee (3D FCN), "
                         "chen (3D CNN), "
                         "li (3D CNN), "
                         "he (3D CNN), "
                         "luo (3D CNN), "
                         "sharma (2D CNN), "
                         "boulch (1D semi-supervised CNN), "
                         "liu (3D semi-supervised CNN), "
                         "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                                               "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=0.1,
                           help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                                                             " (random sampling or disjoint, default: random)",
                           default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                           help="Path to the train ground truth (optional, this "
                                "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                           help="Path to the test set (optional, by default "
                                "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, default=1000, help="Training epochs (optional, if"
                                                                 " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, default=7,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

parser.add_argument('--gan', default=True, help="是否选择利用gan进行小数据增强")
parser.add_argument('--pure_gan', default=False, help="是否选择利用纯gan进行小数据增强")
parser.add_argument('--copy', default=False, help="是否进行数据的简单复制")

args = parser.parse_args()
