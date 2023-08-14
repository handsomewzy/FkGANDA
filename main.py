from __future__ import print_function
from __future__ import division

# Torch
import torch, gc
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# Visualization
import seaborn as sns
import visdom
import utils
import os
from utils import metrics, convert_to_color_, convert_from_color_, \
    display_dataset, display_predictions, explore_spectrums, plot_spectrums, \
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import datasets
import model_data
from Completed_Band_Select import create_total_dataset, create_new_dataset
import KeepGAN

opt = KeepGAN.opt


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

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
# PaviaC IndianPines
parser.add_argument('--dataset', type=str, default='IndianPines', choices=dataset_names,
                    help="Dataset to use."
                         "PaviaC"
                         "IndianPines"
                         "PaviaU"
                         "Botswana"
                    )
parser.add_argument('--model', type=str, default='hamida',
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
group_train.add_argument('--epoch', type=int, default=100, help="Training epochs (optional, if"
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

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")
parser.add_argument('--gan', default=False, help="是否选择利用gan进行小数据增强")
parser.add_argument('--pure_gan', default=False, help="是否选择利用纯gan进行小数据增强")
parser.add_argument('--copy', default=False, help="是否进行数据的简单复制")
args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)

# -----------------------------------
# 使用GAN生成所需的小样本数据
# -----------------------------------
img, gt, label_value, ignored_value, rgb_bands, pattle = datasets.get_dataset(dataset_name=opt.dataset_name,
                                                                              target_folder="Datasets"
                                                                              )

train_gt, test_gt = utils.sample_gt(gt, opt.training_sample, mode='random')

copy_signal = False
gan_signal = False

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                                        FOLDER)
# Number of classes
# N_CLASSES = len(LABEL_VALUES) - len(IGNORED_LABELS)
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
print(N_BANDS)

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update(
    {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                       ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode='random')
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    if MODEL == 'SVM_grid':
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SVM':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        # print(X_train.shape)
        # X_train = np.concatenate((X_train, X_train), axis=0)
        # y_train = np.concatenate((y_train, y_train), axis=0)
        # print(X_train.shape)

        # if args.gan:
        #     for img,label in MyDataset_1:
        #         # 取出中间通道的数据，加入训练集
        #
        #         img = img.detach().numpy()
        #         img = img.transpose(0, 3, 1,2)
        #
        #         label = label.detach().numpy()
        #         X_train += list(img[0][3][3])
        #         y_train += list[label]

        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        # save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3,
                                                 average=10)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'nearest':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.neighbors.KNeighborsClassifier(weights='distance')
        clf = sklearn.model_selection.GridSearchCV(clf, {'n_neighbors': [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    else:
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

        # Split train set in train/val
        # train_gt, val_gt = sample_gt(train_gt, 0.9, mode='random')

        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)

        # ————————————————————————————————
        # 对小样本进行复制，查看如果完美生成是否会有改进
        # ————————————————————————————————
        # copy_dataset = datasets.HyperX_choose(img, train_gt, patch_size=7, choose_labels=[1],
        #                                       center_pixel=True, supervision='full')

        # ————————————————————————————————
        # 加入生成器生成的样本数据，优化结果
        # ————————————————————————————————
        if args.gan:
            print("扩增前的训练集的长度为：{}".format(len(train_dataset)))
            # KSC
            # num_list = [38, 12, 13, 13, 8, 11, 5, 22, 26, 20, 21, 25, 46]

            # Botswana
            # num_list = [14, 5, 13, 11, 13, 13, 13, 10, 16, 12, 15, 9, 13, 5]

            # IP 1 3 4 5 7 9 10 12 13 15
            num_list = [5, 143, 72, 24, 41, 73, 3, 48, 2, 96, 236, 59, 20, 126, 26]
            # num_list = [5, 72, 24, 41, 3, 2, 96, 59, 20, 26]

            # PU
            # num_list = [66, 186, 20, 30, 13, 50, 13, 36, 9]
            # num_list = [66, 186, 20, 13, 50, 13, 36]

            # for i in range(len(num_list)):
            #     num_list[i] = num_list[i] // 2
            print(num_list)

            new_train_dataset = create_total_dataset(origin_dataset=train_dataset, num_list=num_list,
                                                     mix_percent=1)

            # new_train_dataset = train_dataset + create_new_dataset(pth="gan_5_Botswana_0.050000.pth"
            #                                                        , label=5, num=13, percent=0.6) + create_new_dataset(
            #     pth="gan_6_Botswana_0.050000.pth"
            #     , label=6, num=13, percent=0.6)

            print("扩增后的训练集的长度为：{}".format(len(new_train_dataset)))
            train_loader = data.DataLoader(new_train_dataset,
                                           batch_size=hyperparams['batch_size'],
                                           pin_memory=False,
                                           shuffle=True)
            print("训练集的长度为：%d" % len(train_loader))

            gc.collect()
            torch.cuda.empty_cache()
        else:
            train_loader = data.DataLoader(train_dataset,
                                           batch_size=hyperparams['batch_size'],
                                           pin_memory=False,
                                           shuffle=True)
            print("训练集的长度为：%d" % len(train_loader))

        # val_dataset = HyperX(img, val_gt, **hyperparams)
        # val_loader = data.DataLoader(val_dataset,
        #                              pin_memory=False,
        #                              batch_size=hyperparams['batch_size'])
        # print("验证集的长度为：%d" % len(val_loader))

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            print(input.size()[1:])
            # summary(model.to(hyperparams['device']), input.size()[1:], device=hyperparams['device'])
            # summary(model.to(hyperparams['device']), torch.Size([200,7,7]))
            summary(model.to(hyperparams['device']), input.size()[1:])

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'], val_loader=None,
                  display=viz)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    # run_results = metrics(prediction, test_gt, ignored_labels=[0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17]
    #                       , n_classes=4)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, viz, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)

    # show_results(run_results, viz, label_values=["Undefined", "Alfalfa", "Grass-pasture-mowed", "Oats"])

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
