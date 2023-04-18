# system, numpy
import os
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

# user defined
import utils


class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(4096, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Generator(nn.Module):
    # 512->25,344->25,344->(1*224*224)
    def __init__(self, in_dim=512, out_dim=300, num_clss=10, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_clss, num_clss)
        # hid_dim = int((in_dim + out_dim) / 2)
        # modules = list()
        # modules.append(nn.Linear(in_dim, hid_dim))
        # if use_batchnorm:
        #     modules.append(nn.BatchNorm1d(hid_dim))
        # modules.append(nn.LeakyReLU(0.2, inplace=True))
        # if noise:
        #     modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        # if use_dropout:
        #     modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.Linear(hid_dim, hid_dim))
        # if use_batchnorm:
        #     modules.append(nn.BatchNorm1d(hid_dim))
        # modules.append(nn.LeakyReLU(0.2, inplace=True))
        # if noise:
        #     modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        # if use_dropout:
        #     modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.Linear(hid_dim, out_dim))
        # self.gen = nn.Sequential(*modules)

        def block():
            layers = [nn.Linear(in_dim, out_dim)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.gen = nn.Sequential(
            *block(in_dim + num_clss, 128, use_batchnorm=False)
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(1, out_dim, out_dim))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_in = torch.cat((self.label_emb(labels), noise), -1)
        img = self.gen(gen_in)
        img = img.view(img.size(0), )
        return img


class Discriminator(nn.Module):
    def __init__(self, in_dim=300, out_dim=1, num_clss=10, noise=True, use_batchnorm=True, use_dropout=False,
                 use_sigmoid=False):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_clss, num_clss)
        # hid_dim = int(in_dim / 2)
        # modules = list()
        # if noise:
        #     modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        # modules.append(nn.Linear(in_dim, hid_dim))
        # if use_batchnorm:
        #     modules.append(nn.BatchNorm1d(hid_dim))
        # modules.append(nn.LeakyReLU(0.2, inplace=True))
        # if use_dropout:
        #     modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.Linear(hid_dim, hid_dim))
        # if use_batchnorm:
        #     modules.append(nn.BatchNorm1d(hid_dim))
        # modules.append(nn.LeakyReLU(0.2, inplace=True))
        # if use_dropout:
        #     modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.Linear(hid_dim, out_dim))
        # if use_sigmoid:
        #     modules.append(nn.Sigmoid())
        #
        # self.disc = nn.Sequential(*modules)
        self.disc = nn.Sequential(
            nn.Linear(num_clss + int(np.prod(1, out_dim, out_dim))),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.disc(x)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.cuda()
            x = x + noise
        return x


class cWGAN(nn.Module):
    def __init__(self, params_model):
        super(cWGAN, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding and semantic embedding
        self.dim_out = params_model['dim_out']
        self.sem_dim = params_model['sem_dim']
        # Number of classes
        self.num_clss = params_model['num_clss']
        # Sketch model: pre-trained on ImageNet
        self.sketch_model = VGGNetFeats(pretrained=False, finetune=False)   # 预训练要改成True吗？
        self.load_weight(self.sketch_model, params_model['path_sketch_model'], 'sketch')
        # Image model: pre-trained on ImageNet
        self.image_model = VGGNetFeats(pretrained=False, finetune=False)
        self.load_weight(self.image_model, params_model['path_image_model'], 'image')
        # Semantic model embedding
        self.sem = []
        for f in params_model['files_semantic_labels']:
            self.sem.append(np.load(f, allow_pickle=True).item())
        self.dict_clss = params_model['dict_clss']
        print('Done')

        print('Initializing trainable models...', end='')
        # Generators
        # Sketch_generator
        self.gen_sk = Generator(in_dim=512, out_dim=self.dim_out, num_clss=self.num_clss, noise=True, use_dropout=True)
        # Image_generator
        self.gen_im = Generator(in_dim=512, out_dim=self.dim_out, num_clss=self.num_clss, noise=True, use_dropout=True)
        # Discriminators
        # Sketch_discriminator
        self.disc_sk = Discriminator(in_dim=self.dim_out, noise=True, use_dropout=True)
        # Image_discriminator
        self.disc_im = Discriminator(in_dim=self.dim_out, noise=True, use_dropout=True)
        # Semantic encoder

        # Optimizers

    def load_weight(self, model, path, type='sketch'):
        checkpoint = torch.load(os.path.join(path, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict_' + type])

    def forward(self, sk, im, se):
        self.sk_fe = self.sketch_model(sk)
        self.im_fe = self.image_model(im)


