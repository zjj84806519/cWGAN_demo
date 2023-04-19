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
    def __init__(self, in_dim=300, out_dim=224*224, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator, self).__init__()

        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        self.gen = nn.Sequential(*modules)
        # self.label_emb = nn.Embedding(num_clss, num_clss)
        #
        # def block():
        #     layers = [nn.Linear(in_dim, out_dim)]
        #     if use_batchnorm:
        #         layers.append(nn.BatchNorm1d(out_dim))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers
        #
        # self.gen = nn.Sequential(
        #     *block(in_dim + num_clss, 128, use_batchnorm=False)
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(1, out_dim, out_dim))),
        #     nn.Tanh()
        # )

    def forward(self, sem):
        img = self.gen(sem)
        img = img.view(-1, 224, 224, 1)
        return img


class Discriminator(nn.Module):
    # 300 -> 150 -> 150 -> 1
    def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False,
                 use_sigmoid=False):
        super(Discriminator, self).__init__()

        hid_dim = int(in_dim / 2)
        modules = list()
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.disc = nn.Sequential(*modules)
        # self.label_emb = nn.Embedding(num_clss, num_clss)
        # self.disc = nn.Sequential(
        #     nn.Linear(num_clss + int(np.prod(1, out_dim, out_dim))),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1)
        # )

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


class cWGan(nn.Module):
    def __init__(self, params_model):
        super(cWGan, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding and semantic embedding
        self.dim_out = params_model['dim_out']
        self.sem_dim = params_model['sem_dim']
        # Number of classes
        self.num_clss = params_model['num_clss']
        # Parameters
        self.noise = torch.FloatTensor()

        # Sketch model: pre-trained on ImageNet
        self.sketch_model = VGGNetFeats(pretrained=False, finetune=False)   # 预训练要改成True吗？
        # self.load_weight(self.sketch_model, params_model['path_sketch_model'], 'sketch')
        # Image model: pre-trained on ImageNet
        self.image_model = VGGNetFeats(pretrained=False, finetune=False)
        # self.load_weight(self.image_model, params_model['path_image_model'], 'image')

        # Semantic model embedding
        self.sem = []
        for f in params_model['files_semantic_labels']:
            self.sem.append(np.load(f, allow_pickle=True).item())
        self.dict_clss = params_model['dict_clss']
        print('Done')

        print('Initializing trainable models...', end='')
        # Generators
        # Sketch_generator
        self.gen_sk = Generator(in_dim=512, out_dim=self.dim_out)
        # Image_generator
        self.gen_im = Generator(in_dim=512, out_dim=self.dim_out)
        # Discriminators
        # Sketch_discriminator
        self.disc_sk = Discriminator(in_dim=self.dim_out)
        # Image_discriminator
        self.disc_im = Discriminator(in_dim=self.dim_out)
        # Semantic encoder(Word2Vec)

        # Photo_Sketching

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.gamma = params_model['gamma']
        self.momentum = params_model['momentum']
        self.milestones = params_model['milestones']
        self.optimizer_gen = optim.Adam(list(self.gen_sk.parameters()) + list(self.gen_im.parameters()), lr=self.lr)
        self.optimizer_disc = optim.SGD(list(self.disc_sk.parameters()) + list(self.disc_im.parameters()), lr=self.lr,
                                        momentum=self.momentum)
        self.scheduler_gen = optim.lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=self.milestones,
                                                            gamma=self.gamma)
        self.scheduler_disc = optim.lr_scheduler.MultiStepLR(self.optimizer_disc, milestones=self.milestones,
                                                             gamma=self.gamma)
        print('Done')

        # loss function
        print('Defining losses...', end='')
        self.criterion_gan = nn.MSELoss()
        print('Done')

        # Initialize variables
        print('Initializing variables...', end='')
        self.sk_fe = torch.zeros(1)
        self.im_fe = torch.zeros(1)
        self.se_fe = torch.zeros(1)
        self.fake_sk = torch.zeros([self.dim_out, self.dim_out])
        self.fake_im = torch.zeros([self.dim_out, self.dim_out])
        self.fake_sk_fe = torch.zeros(1)
        self.fake_im_fe = torch.zeros(1)
        print('Done')

    def load_weight(self, model, path, type='sketch'):
        checkpoint = torch.load(os.path.join(path, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict_' + type])

    def forward(self, sk, im, se):
        self.sk_fe = self.sketch_model(sk)
        self.im_fe = self.image_model(im)

        # Generate fake example with generators
        self.fake_sk = self.gen_sk(se)
        self.fake_im = self.gen_im(se)

        # transform fake image to sketch with PhotoSketching
        # self.im_sk =
        # Transform fake example to embedding
        self.fake_sk_fe = self.sketch_model(self.fake_sk)
        self.fake_sk_fe = self.image_model(self.fake_im)

    def backward(self):

        # Generator loss
        loss_gen = self.criterion_gan(self.disc_sk(self.fake_sk_fe)) + self.criterion_gan(self.disc_im(self.fake_im_fe))
        # Weighted loss_gen = self.lambda_gen * loss_gen
        # initialize optimizer for generator
        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        # Optimizer step
        self.optimizer_gen.step()

        # initialize optimizer for discriminator
        self.optimizer_disc.zero_grad()
        # Sketch discriminator loss
        loss_disc_sk = self.criterion_gan(self.disc_sk(self.sk_fe)) + self.criterion_gan(self.disc_sk(self.fake_sk_fe))
        # Weighted loss_disc_sk = self.lambda_disc_sk * loss_disc_sk
        loss_disc_sk.backward(retain_graph=True)

        # Image discriminator loss
        loss_disc_im = self.criterion_gan(self.disc_im(self.im_fe)) + self.criterion_gan(self.disc_im(self.fake_im_fe))
        # Weighted loss_disc_im = self.lambda_disc_im * loss_disc_im
        loss_disc_im.backward(retain_graph=True)
        # Optimizer step
        self.optimizer_disc.step()

        losses_disc = loss_disc_sk + loss_disc_im

        loss = {'gen_loss': loss_gen, 'disc_sk': loss_disc_sk, 'disc_im': loss_disc_im, 'disc': losses_disc}

        return loss

    def optimize_params(self, sk, im, cl):
        # Get numeric classes
        num_cls = torch.from_numpy(utils.create_dict_texts(cl, self.dict_clss)).cuda()

        # Get the semantic embedding for cl
        se = np.zeros((len(cl), self.sem_dim), dtype=np.float32)
        for i, c in enumerate(cl):
            se_c = np.array([], dtype=np.float32)
            for s in self.sem:
                se_c = np.concatenate((se_c, s.get(c).astype(np.float32)), axis=0)
            se[i] = se_c
        se = torch.from_numpy(se)
        if torch.cuda.is_available:
            se = se.cuda()

        # Forward pass
        self.forward(sk, im, se)

        # Backward pass
        loss = self.backward()

        return loss




