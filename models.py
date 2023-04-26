# system, numpy
import os
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as F
from PIL import Image


# user defined
import utils
from losses import GANLoss

device = torch.device("cuda:2")


class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 5, 4096),
            *list(model.classifier.children())[1:-1],
            nn.Linear(4096, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Generator(nn.Module):
    # 300>25,344->25,344->(3*224*224)
    def __init__(self, in_dim=300, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
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
        # modules.append(nn.Tanh())
        self.gen = nn.Sequential(*modules)
        for idx, m in enumerate(self.gen.modules()):
            print(idx, '->', m)
        # num_clss = 32
        # self.label_emb = nn.Embedding(num_clss, num_clss)
        #
        # def block(in_feat, out_feat, use_batchnorm=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     if use_batchnorm:
        #         layers.append(nn.BatchNorm1d(out_feat, 0.8))
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
        print('gen done')

    def forward(self, se):
        img = self.gen(se)      # [32, 50176(224*224)]
        # print("before image size:", img.size())
        img = img.view(img.size(0), 1, 160, 160).repeat(1, 3, 1, 1)     # 使用 repeat() 方法将单通道张量复制三遍，变成三通道张量
        # print("after image size:", img.size())
        # pil_img = (img * 255).clamp(0, 255).to(torch.uint8)     # 将张量的值从 [0,1] 转换为 [0,255] 的整数
        # print("final image size:", pil_img.size())
        return img


class Discriminator(nn.Module):
    # 512 -> 256 -> 256 -> 1
    def __init__(self, in_dim=512, out_dim=1, noise=False, use_batchnorm=True, use_dropout=False,
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
        for idx, m in enumerate(self.disc.modules()):
            print(idx, '->', m)
        print('disc done')
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

    def forward(self, img):
        # img = img.view(img.size(0), -1)
        # print("disc img size:", img.size())
        return self.disc(img)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.to(device)
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
        self.img_shape = params_model['img_shape']

        # Sketch model: pre-trained on ImageNet
        self.sketch_model = VGGNetFeats(pretrained=True, finetune=False)
        # self.load_weight(self.sketch_model, params_model['path_sketch_model'], 'sketch')
        # Image model: pre-trained on ImageNet
        self.image_model = VGGNetFeats(pretrained=True, finetune=False)
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
        self.gen_sk = Generator(in_dim=300, out_dim=self.dim_out)
        # Image_generator
        self.gen_im = Generator(in_dim=300, out_dim=self.dim_out)
        # Discriminators
        # Sketch_discriminator
        self.disc_sk = Discriminator(in_dim=512, use_sigmoid=False)
        # Image_discriminator
        self.disc_im = Discriminator(in_dim=512, use_sigmoid=False)
        # Semantic encoder(Word2Vec)

        # Photo_Sketching

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.gamma = params_model['gamma']
        self.momentum = params_model['momentum']
        self.milestones = params_model['milestones']
        self.channels = params_model['channels']
        self.clip_value = params_model['clip_value']
        self.optimizer_gen = optim.Adam(list(self.gen_sk.parameters()) + list(self.gen_im.parameters()), lr=self.lr)
        # self.optimizer_gen = optim.SGD(list(self.gen_sk.parameters()) + list(self.gen_im.parameters()), lr=self.lr,
        #                               momentum=self.momentum)
        self.optimizer_disc = optim.SGD(list(self.disc_sk.parameters()) + list(self.disc_im.parameters()), lr=self.lr,
                                        momentum=self.momentum)
        self.scheduler_gen = optim.lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=self.milestones,
                                                            gamma=self.gamma)
        self.scheduler_disc = optim.lr_scheduler.MultiStepLR(self.optimizer_disc, milestones=self.milestones,
                                                             gamma=self.gamma)
        print('Done')

        # loss function
        print('Defining losses...', end='')
        self.criterion_gan = GANLoss(use_lsgan=True)
        self.lambda_gen = params_model['lambda_gen']
        self.lambda_disc_sk = params_model['lambda_disc_sk']
        self.lambda_disc_im = params_model['lambda_disc_im']
        print('Done')

        # Initialize variables
        print('Initializing variables...', end='')
        self.sk_fe = torch.zeros(1)
        self.im_fe = torch.zeros(1)
        self.se_fe = torch.zeros(1)
        self.fake_sk = torch.zeros(self.img_shape)  # (3, 128, 128)
        self.fake_im = torch.zeros(self.img_shape)
        self.fake_sk_fe = torch.zeros(1)
        self.fake_im_fe = torch.zeros(1)
        self.fake_sk_fe_g = torch.zeros(1)
        self.fake_im_fe_g = torch.zeros(1)
        print('Done')

    def load_weight(self, model, path, type='sketch'):
        checkpoint = torch.load(os.path.join(path, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict_' + type])

    def forward(self, sk, im, se):
        # print("sketch的尺寸：{},type:{}，image的尺寸：{},type:{}".format(sk.size(), type(sk), im.size(), type(im)))
        # [batch_size, 3, 224, 224]
        self.sk_fe = self.sketch_model(sk)
        self.im_fe = self.image_model(im)

        # print("sketch fe的尺寸：{}，image fe的尺寸：{}".format(self.sk_fe.size(), self.im_fe.size()))  # [batch_size, 512]
        # Generate fake example with generators and transform to embedding
        self.fake_sk_fe = self.sketch_model(self.gen_sk(se))
        self.fake_im_fe = self.image_model(self.gen_im(se))
        self.fake_sk_fe_g = self.fake_sk_fe.clone()    # 固定
        self.fake_im_fe_g = self.fake_im_fe.clone()
        # print("fake sketch fe的尺寸：{}，fake image fe的尺寸：{}".format(self.fake_sk_fe.size(), self.fake_im_fe.size()))

        return self.gen_sk(se), self.gen_im(se)

    def backward(self):

        # Generator loss
        # loss_gen = self.criterion_gan(self.disc_sk(self.fake_sk_fe), True) + \
        #            self.criterion_gan(self.disc_im(self.fake_im_fe), True)
        # loss_gen = self.lambda_gen * loss_gen
        # # initialize optimizer for generator
        # self.optimizer_gen.zero_grad()
        # loss_gen.backward(retain_graph=True)
        # # Optimizer step
        # self.optimizer_gen.step()
        #
        # # initialize optimizer for discriminator
        # self.optimizer_disc.zero_grad()
        # # Sketch discriminator loss
        # loss_disc_sk = self.criterion_gan(self.disc_sk(self.sk_fe), True) +\
        #                self.criterion_gan(self.disc_sk(self.fake_sk_fe), False)
        # loss_disc_sk = self.lambda_disc_sk * loss_disc_sk
        # loss_disc_sk.backward(retain_graph=True)
        #
        # # Image discriminator loss
        # loss_disc_im = self.criterion_gan(self.disc_im(self.im_fe), True) + \
        #                self.criterion_gan(self.disc_im(self.fake_im_fe), False)
        # loss_disc_im = self.lambda_disc_im * loss_disc_im
        # loss_disc_im.backward(retain_graph=True)
        # # Optimizer step
        # self.optimizer_disc.step()
        #
        # losses_disc = loss_disc_sk + loss_disc_im

        # Adversarial loss

        # 将判别器的参数设置为需要梯度
        # for p in list(self.disc_sk.parameters()) + list(self.disc_im.parameters()):
        #     p.requires_grad = True

        self.optimizer_disc.zero_grad()

        loss_disc_sk = -torch.mean(self.disc_sk(self.sk_fe)) + torch.mean(self.disc_sk(self.fake_sk_fe_g.detach()))
        # loss_disc_sk = self.lambda_disc_sk * loss_disc_sk
        loss_disc_sk.backward(retain_graph=True)

        loss_disc_im = -torch.mean(self.disc_im(self.sk_fe)) + torch.mean(self.disc_sk(self.fake_im_fe_g.detach()))
        # loss_disc_im = self.lambda_disc_im * loss_disc_im
        loss_disc_im.backward(retain_graph=True)

        self.optimizer_disc.step()

        losses_disc = loss_disc_sk + loss_disc_sk

        # 固定判别器的参数
        # for p in list(self.disc_sk.parameters()) + list(self.disc_im.parameters()):
        #     p.requires_grad = False

        # 截断判别器的loss
        for p in list(self.disc_sk.parameters()) + list(self.disc_im.parameters()):
            p.data.clamp_(-self.clip_value, self.clip_value)    # (-0.01, 0.01)

        # Generator loss
        self.optimizer_gen.zero_grad()

        loss_gen_sk = -torch.mean(self.disc_sk(self.fake_sk_fe))
        loss_gen_sk.backward(retain_graph=True)

        loss_gen_im = -torch.mean(self.disc_im(self.fake_im_fe))
        loss_gen_im.backward(retain_graph=True)

        self.optimizer_gen.step()

        loss_gen = loss_gen_sk + loss_gen_im

        loss = {'gen_loss': loss_gen, 'disc_sk': loss_disc_sk, 'disc_im': loss_disc_im, 'disc': losses_disc}

        return loss

    def optimize_params(self, sk, im, cl):
        # Get numeric classes
        # num_cls = torch.from_numpy(utils.create_dict_texts(cl, self.dict_clss)).to(device)

        # Get the semantic embedding for cl
        se = np.zeros((len(cl), self.sem_dim), dtype=np.float32)    # (32, 300)
        for i, c in enumerate(cl):
            se_c = np.array([], dtype=np.float32)
            for s in self.sem:
                se_c = np.concatenate((se_c, s.get(c).astype(np.float32)), axis=0)
            se[i] = se_c
        se = torch.from_numpy(se)
        if torch.cuda.is_available:
            se = se.to(device)  # dim=300
        # Forward pass
        fake_sk, fake_im = self.forward(sk, im, se)

        # Backward pass
        loss = self.backward()

        return loss, fake_sk, fake_im

    def get_sketch_embeddings(self, se):
        # sketch embedding
        sk_em = self.gen_sk(se)

        return sk_em

    def get_image_embeddings(self, se):
        # image embedding
        im_em = self.gen_im(se)

        return im_em