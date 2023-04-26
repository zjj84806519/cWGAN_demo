"""
    Parse input arguments
"""

import utils
import argparse


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='cWGAN demo for Zero-Shot Sketch-based Image Retrieval')
        # Optional argument
        parser.add_argument('--dataset', default='TU-Berlin', help='Name of the dataset')
        # Different training test sets
        parser.add_argument('--gzs-sbir', action='store_true', default=False,
                            help='Generalized zero-shot sketch based image retrieval')
        parser.add_argument('--filter-sketch', action='store_true', default=False, help='Allows only one sketch per '
                                                                                        'image (only for Sketchy)')
        # Pretrained models
        parser.add_argument('--semantic-models', nargs='+', default=['word2vec-google-news'],   # , 'hieremb-path'
                            type=str, help='Path to the semantic model')
        parser.add_argument('--photo-sketch', default='./pretrained/photosketch.pth', type=str,
                            help='Path to the photosketch pre-trained model')
        # Weight parameters
        parser.add_argument('--lambda-gen', default=1.0, type=float, help='Weight on adversarial loss (gen)')
        parser.add_argument('--lambda-disc-sk', default=1.0, type=float, help='Weight on sketch loss (disc)')
        parser.add_argument('--lambda-disc-im', default=1.0, type=float, help='Weight on image loss (disc)')
        # Size parameters
        parser.add_argument('--channels', default=3, type=int, help='number of image channels')
        parser.add_argument('--im-sz', default=160, type=int, help='Image size')
        parser.add_argument('--sk-sz', default=160, type=int, help='Sketchy size')
        parser.add_argument('--dim-out', default=160*160, type=int, help='Output dimension of sketch and image')
        # Model parameters
        parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
        parser.add_argument('--epoch-size', default=100, type=int, help='Epoch size')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in data loader')
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--clip-value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
        # Checkpoint parameters
        parser.add_argument('--test', action='store_true', default=False, help='Test only flag')
        # Optimization parameters
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='Number of epochs to train (default: 100)')
        parser.add_argument('--lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR',
                            help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--milestones', type=int, nargs='+', default=[], help='Milestones for scheduler')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule steps.')
        # I/O parameters
        parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument("--sample-interval", type=int, default=100, help="interval betwen image samples")
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()