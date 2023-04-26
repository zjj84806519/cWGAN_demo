# system, numpy
import os
import time
import numpy as np
from scipy.spatial.distance import cdist

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# user defined
# import itq
import utils
from options import Options
from logger import Logger, AverageMeter
from models import cWGan
from data import DataGeneratorSketch, DataGeneratorImage

device = torch.device("cuda:2")
np.random.seed(0)


def main():
    print()


def validate(valid_loader_sketch, valid_loader_image, cwgan_model, epoch, args):

    # Switch to test mode
    cwgan_model.eval()

    batch_time = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, cls_sk) in enumerate(valid_loader_sketch):
        if torch.cuda.is_available():
            sk = sk.to(device)

            # Sketch embedding into a semantic space
            sk_em = cwgan_model.get_sketch_embeddings(sk)

    # for i, (im, cls_im) in enumerate(valid_loader_image):
