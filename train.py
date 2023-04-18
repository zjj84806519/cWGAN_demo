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
from models import cWGAN
from logger import Logger, AverageMeter
from options import Options
from data import DataGeneratorPaired, DataGeneratorSketch, DataGeneratorImage


def main():
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Path
    path_dataset = 'F:\\PythonWorkPlace\\cWGAN_demo'
    path_aux = 'F:\\JUN\\Academic\\Research\\Dataset\\SBIR'

    # modify the log and checkpoint paths
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
    # Optimizor
    params_model['lr'] = args.lr
    params_model['momentum'] = args.momentum
    params_model['milestones'] = args.milestones
    params_model['gamma'] = args.gamma

    # Model
    cwgan_model = cWGAN(params_model)

    # Logger
    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        cwgan_model = cwgan_model.cuda()
    print('Done')

    # Epoch for loop
    if not args.test:
        print('***Train***')
        for epoch in range(args.epochs):

            cwgan_model.scheduler_gen.step()
            cwgan_model.scheduler_disc.step()

        # train on training set
        losses = train(train_loader, cwgan_model, epoch, args)

        # evaluate on validation set, map_ since map is already there
        # print('***Validation***')
        # valid_data = validate(valid_loader_sketch, valid_loader_image, sem_pcyc_model, epoch, args)

        # Logger step
        logger.add_scalar('generator loss', losses['gen_loss'].avg)
        logger.add_scalar('sketch discriminator loss', losses['disc_sk'].avg)
        logger.add_scalar('image discriminator loss', losses['disc_im'].avg)
        logger.add_scalar('discriminator loss', losses['disc'].avg)


def train(train_loader, cwgan_model, epoch, args):
    # Switch to train mode
    cwgan_model.train()

    batch_time = AverageMeter()
    loss_gen = AverageMeter()
    loss_disc_sk = AverageMeter()
    loss_disc_im = AverageMeter()
    losses_disc = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, im, cl) in enumerate(train_loader):
        # Transfer sk and im to cuda
        if torch.cuda.is_available:
            sk, im = sk.cuda(), im.cuda()

        # Optimize parameters
        loss = cwgan_model.optimize_params(sk, im, cl)

        # Store losses for visualization
        loss_gen.update(loss['gen_loss'].item(), sk.size(0))
        loss_disc_sk.update(loss['disc_sk'].item(), sk.size(0))
        loss_disc_im.update(loss['disc_im'].item(), sk.size(0))
        losses_disc.update(loss['disc'].item(), sk.size(0))
        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Gen. Loss {loss_gen.val:.4f} ({loss_gen.avg:.4f})\t'
                  'Disc. Loss {loss_disc.val:.4f} ({loss_disc.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss_gen=loss_gen,
                          loss_disc=losses_disc))

        losses = {'gen_loss': loss_gen, 'disc_sk': loss_disc_sk, 'disc_im': loss_disc_im, 'disc': losses_disc}

    return losses


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
