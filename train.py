# system, numpy
import os
import time
import numpy as np

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

# user defined
import utils
import models
from options import Options
from data import DataGeneratorPaired, DataGeneratorSketch, DataGeneratorImage


def main():
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Path
    path_dataset = 'F:\\PythonWorkPlace\\cWGAN_demo'
    path_aux = 'F:\\JUN\\Academic\\Research\\Dataset\\SBIR'

    # modify
    ds_var = None
    if '_' in args.dataset:
        token = args.dataset.split('_')
        args.dataset = token[0]
        ds_var = token[1]

    str_aux = ''
    if args.gzs_sbir:
        str_aux = os.path.join(str_aux, 'generalized')
    #
    args.semantic_models = sorted(args.semantic_models)
    model_name = '+'.join(args.semantic_models)
    #
    root_path = os.path.join(path_dataset, args.dataset)
    path_sketch_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'sketch')
    path_image_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'image')
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out))
    path_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    #
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]
    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_results))
    #
    # Parameters for transforming the images
    trans_image = transforms.Compose([transforms.Resize((args.im_sz, args.im_sz)), transforms.ToTensor()])
    trans_sketch = transforms.Compose([transforms.Resize((args.sk_sz, args.sk_sz)), transforms.ToTensor()])

    # Load the dataset
    print('Loading data...', end='')
    if args.dataset == 'Sketchy':
        if ds_var == 'extended':
            photo_dir = 'extended_photo'  # photo or extended_photo
            photo_sd = ''
        else:
            photo_dir = 'photo'
            photo_sd = 'tx_000000000000'
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000'

    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        splits = utils.load_files_tuberlin_zeroshot(root_path=root_path, photo_dir=photo_dir, sketch_dir=sketch_dir,
                                                    photo_sd=photo_sd, sketch_sd=sketch_sd)
    else:
        raise Exception('Wrong dataset.')
    # Combine the valid and test set into test set
    splits['te_fls_sk'] = np.concatenate((splits['va_fls_sk'], splits['te_fls_sk']), axis=0)
    splits['te_clss_sk'] = np.concatenate((splits['va_clss_sk'], splits['te_clss_sk']), axis=0)
    splits['te_fls_im'] = np.concatenate((splits['va_fls_im'], splits['te_fls_im']), axis=0)
    splits['te_clss_im'] = np.concatenate((splits['va_clss_im'], splits['te_clss_im']), axis=0)

    # Generalized Zero-Shot SBIR
    if args.gzs_sbir:
        perc = 0.2
        _, idx_sk = np.unique(splits['tr_fls_sk'], return_index=True)
        tr_fls_sk_ = splits['tr_fls_sk'][idx_sk]
        tr_clss_sk_ = splits['tr_clss_sk'][idx_sk]
        _, idx_im = np.unique(splits['tr_fls_im'], return_index=True)
        tr_fls_im_ = splits['tr_fls_im'][idx_im]
        tr_clss_im_ = splits['tr_clss_im'][idx_im]
        if args.dataset == 'Sketchy' and args.filter_sketch:
            _, idx_sk = np.unique([f.split('-')[0] for f in tr_fls_sk_], return_index=True)
            tr_fls_sk_ = tr_fls_sk_[idx_sk]
            tr_clss_sk_ = tr_clss_sk_[idx_sk]
        idx_sk = np.sort(np.random.choice(tr_fls_sk_.shape[0], int(perc * splits['te_fls_sk'].shape[0]), replace=False))
        idx_im = np.sort(np.random.choice(tr_fls_im_.shape[0], int(perc * splits['te_fls_im'].shape[0]), replace=False))
        splits['te_fls_sk'] = np.concatenate((tr_fls_sk_[idx_sk], splits['te_fls_sk']), axis=0)
        splits['te_clss_sk'] = np.concatenate((tr_clss_sk_[idx_sk], splits['te_clss_sk']), axis=0)
        splits['te_fls_im'] = np.concatenate((tr_fls_im_[idx_im], splits['te_fls_im']), axis=0)
        splits['te_clss_im'] = np.concatenate((tr_clss_im_[idx_im], splits['te_clss_im']), axis=0)

    # class dictionary
    dict_clss = utils.create_dict_texts(splits['tr_clss_im'])

    data_train = DataGeneratorPaired(args.dataset, root_path, photo_dir, sketch_dir, photo_sd, sketch_sd,
                                     splits['tr_fls_sk'], splits['tr_fls_im'], splits['tr_clss_im'],
                                     transforms_sketch=trans_sketch, transforms_image=trans_image)
    data_valid_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['va_fls_sk'],
                                            splits['va_clss_sk'], transforms=trans_sketch)
    data_valid_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['va_fls_im'],
                                          splits['va_clss_im'], transforms=trans_image)
    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=trans_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_im'],
                                         splits['te_clss_im'], transforms=trans_image)
    print('Done')

    train_sampler = WeightedRandomSampler(data_train.get_weights(), num_samples=args.epoch_size * args.batch_size,
                                          replacement=True)

    # PyTorch train loader
    train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    # PyTorch valid loader for sketch

    # PyTorch valid loader for image

    # PyTorch test loader for sketch

    # PyTorch test loader for image

    # Model parameters

    print('DataLoader Done!')

    params_model = dict()
    # Paths to pre-trained sketch and image models
    params_model['path_sketch_model'] = path_sketch_model
    params_model['path_image_model'] = path_image_model
    # Dimensions
    params_model['dim_out'] = args.dim_out
    #
    params_model['sem_dim'] = sem_dim
    #
    # Number of classes
    params_model['num_clss'] = len(dict_clss)
    params_model['img_shape'] = (1, args.im_sz, args.sk_sz)
    # Weight (on losses) parameters

    # Optimizers' parameters

    # Files with semantic labels
    params_model['files_semantic_labels'] = files_semantic_labels
    # Class dictionary
    params_model['dict_clss'] = dict_clss

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netG = models.Generator().to(device)
    netD = models.Discriminator().to(device)

    # Optimizor
    

    # epoch for loop

    #


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
