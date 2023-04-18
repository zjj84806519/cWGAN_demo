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
        parser.add_argument('--dataset', required=True, default='TU-Berlin', help='Name of the dataset')
        # Different training test sets
        parser.add_argument('--gzs-sbir', action='store_true', default=False,
                            help='Generalized zero-shot sketch based image retrieval')
        parser.add_argument('--filter-sketch', action='store_true', default=False, help='Allows only one sketch per '
                                                                                        'image (only for Sketchy)')
        # Semantic models
        parser.add_argument('--semantic-models', nargs='+', default=['word2vec-google-news', 'hieremb-path'],
                            type=str, help='Semantic model')
        # Weight parameters
        parser.add_argument('--lambda', default=10.0, type=float, help='Weight on the model')
        # Size parameters
        parser.add_argument('--channels', default=1, type=int, help='number of image channels')
        parser.add_argument('--im-sz', default=224, type=int, help='Image size')
        parser.add_argument('--sk-sz', default=224, type=int, help='Sketchy size')
        parser.add_argument('--dim-out', default=224*224, type=int, help='Output dimension of sketch and image')
        # Model parameters
        parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
        parser.add_argument('--epoch-size', default=100, type=int, help='Epoch size')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in data loader')
        # Checkpoint parameters

        # Optimization parameters
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='Number of epochs to train (default: 100)')
        # I/O parameters

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()