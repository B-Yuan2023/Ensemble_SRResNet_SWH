#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:58:42 2024

@author: g260218
"""

import torch.nn as nn
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


# this can be used to upsample/downsample using interpolation
class Interpolate(nn.Module):
    def __init__(self, size=None,scale_factor=None,mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size,scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


# model srcnn and fsrcnn based on https://github.com/yjn870
class SRCNN(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,up_factor=4):
        super(SRCNN, self).__init__()
        self.conv0 = Interpolate(scale_factor=up_factor, mode='bilinear')  # add a layer to interpolate lr to hr first
        # self.conv0 = nn.Upsample(scale_factor=up_factor,mode='bilinear')  # check if this is the same as the above
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class SRCNN1(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,up_factor=4):
        super(SRCNN1, self).__init__()
        # self.conv0 = Interpolate(scale_factor=up_factor, mode='bilinear')  # add a layer to interpolate lr to hr first
        self.conv0 = nn.Upsample(scale_factor=up_factor,mode='bilinear')  # check if this is the same as the above
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
class FSRCNN(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, up_factor=4, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        upsampling = []
        for i in range(upsample_block_num):
            if i<upsample_block_num-1:
                ochl = d
            else:
                ochl = out_channels
            upsampling += [
                nn.ConvTranspose2d(d, ochl, kernel_size=9, stride=factors[i], padding=9//2,
                                                    output_padding=factors[i]-1),
            ]
        self.last_part = nn.Sequential(*upsampling)
        # self.last_part = nn.ConvTranspose2d(d, out_channels, kernel_size=9, stride=scale_factor, padding=9//2,
        #                                     output_padding=scale_factor-1)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

class FSRCNN1(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, up_factor=4, d=56, s=12, m=4):
        super(FSRCNN1, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        upsampling = []
        for i in range(upsample_block_num):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(d, d*factors[i]**2, 3, 1, 1),
                # nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=factors[i]),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.last_part = nn.Sequential(nn.Conv2d(d, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh())
        # self.last_part = nn.ConvTranspose2d(d, out_channels, kernel_size=9, stride=scale_factor, padding=9//2,
        #                                     output_padding=scale_factor-1)
        # self._initialize_weights()

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.upsampling(x)
        x = self.last_part(x)
        return (x+1)/2 # tanh [-1,1] ->[0,1]

class SRCNN0(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x