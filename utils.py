import argparse
import itertools
import os
import glob
import random

import numpy as np
from matplotlib import pyplot as plt


def create_dict_texts(texts):
    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def get_coarse_grained_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True):
    idx_im_ret = np.array([], dtype=np.int)
    idx_sk_ret = np.array([], dtype=np.int)
    clss_im = np.array([f.split('\\')[-2] for f in fls_im])
    clss_sk = np.array([f.split('\\')[-2] for f in fls_sk])
    names_sk = np.array([f.split('-')[0] for f in fls_sk])
    # print('clss_im={},clss_sk={}'.format(clss_im, clss_sk))

    for i, c in enumerate(classes):
        print(''.format())
        idx1 = np.where(clss_im == c)[0]
        idx2 = np.where(clss_sk == c)[0]
        if set_type == 'train':
            idx_cp = list(itertools.product(idx1, idx2))  # 所有笛卡尔积的列表
            if len(idx_cp) > 100000:  # 截断冗余
                random.seed(i)
                idx_cp = random.sample(idx_cp, 100000)
            idx1, idx2 = zip(*idx_cp)  # 打包成一个个元组的列表
            print('train:第{}次的c={},idx1={},idx2={},idx_cp={}'.format(i, c, len(idx1), len(idx2), len(idx_cp)))
        else:
            # remove duplicate sketches
            if filter_sketch:
                names_sk_tmp = names_sk[idx2]
                idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                idx2 = idx2[idx_tmp]
        idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
        idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)

    return idx_im_ret, idx_sk_ret


def load_files_sketchy_zeroshot(root_path, split_eccv_2018=False, filter_sketch=False, photo_dir='photo',
                                sketch_dir='sketch', photo_sd='tx_000000000000', sketch_sd='tx_000000000000'):
    # paths of sketch and image
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)


def load_files_tuberlin_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd=''):
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    print(path_im, path_sk)
    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = np.array([os.path.join(f.split('\\')[-2], f.split('\\')[-1]) for f in fls_im])
    clss_im = np.array([f.split('\\')[-2] for f in fls_im])

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    # print('before={}'.format(len(fls_sk)))
    fls_sk = np.array([os.path.join(f.split('\\')[-2], f.split('\\')[-1]) for f in fls_sk])
    # print('after={}'.format(len(fls_sk)))
    clss_sk = np.array([f.split('\\')[-2] for f in fls_sk])
    # print('clss_sk={}'.format(len(clss_sk)))
    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes
    np.random.seed(0)
    tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
    va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.1 * len(classes)), replace=False)
    te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train')
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid')
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test')

    splits = dict()
    # splits of sketch files and classes
    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]
    # splits of image files and classes
    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]

    return splits


def gen_sample_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis('off')
    plt.show()
