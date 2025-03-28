import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19,vgg16
import math

def prime_factors(n):
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors


class ResidualBlock(nn.Module):
    def __init__(self, in_features,kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_features), # ,0.8, default momentum 0.1 
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_features), #, 0.8
        )
    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16,up_factor=4,
                 kauxhr=0,auxhr_chls=1):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))

        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        # upsample_block_num = int(math.log(up_factor, 2))
        upsampling = []
        for i in range(upsample_block_num):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 64*factors[i]**2, 3, 1, 1),
                # nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=factors[i]),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.kaux = kauxhr
        self.aux_chls = auxhr_chls
        if self.kaux ==0: # no auxiliary field, original 
            # Final output layer
            self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 
            # self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4) # remove tanh
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            self.conv_aux = nn.Sequential(nn.Conv2d(64+auxhr_chls, 64, 3, 1, 1),nn.PReLU())
            self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 

    def forward(self, x,y=None):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        if self.kaux ==0: # no auxiliary field, original 
            out = self.conv3(out)
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            var = torch.cat((out,y),axis=1)
            out = self.conv_aux(var)
            out = self.conv3(out)
        return (out+1)/2 # tanh [-1,1] ->[0,1]


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # vgg19_model = vgg19(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        # self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        vgg16_model = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:31])
    def forward(self, img):
        return self.feature_extractor(img)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.extend(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_filters, 1024, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1024, 1, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
            ) # follow original structure, fc layer using conv2d

        # layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
