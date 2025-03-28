#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:02:42 2024

@author: g260218
"""

import torch.nn as nn
import torch
import math
from module_attention import SELayer,SELayer1,GCT,CBAM,CoordAtt

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
    
class AttentionBlock(nn.Module):
    def __init__(self, in_features,mod_att=1):
        super(AttentionBlock, self).__init__()
        if mod_att==1:
            self.att_block = SELayer(in_features, reduction=16)
        elif mod_att==2:
            self.att_block = SELayer1(in_features, reduction=16)
        elif mod_att==3:
            self.att_block = GCT(in_features)
        elif mod_att==5:
            self.att_block = CoordAtt(in_features,reduction=16)
        elif mod_att==6:
            self.att_block = CBAM(in_features,reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
    def forward(self, x):
        return self.att_block(x)
    

class ResidualBlockA(nn.Module):
    def __init__(self, in_features,kernel_size=3,mod_att=0):
        super(ResidualBlockA, self).__init__()
        layer_list = [
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_features), # ,0.8, default momentum 0.1 
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_features),
            ]
        if mod_att>0:
            layer_list.append(AttentionBlock(in_features,mod_att)) 
        self.conv_block = nn.Sequential(*layer_list)
    def forward(self, x):
        return x + self.conv_block(x)

# feature extraction before upsampling 
class RS_feature(nn.Module):
    def __init__(self, in_channels,kernel_no=64,kernel_size=3,n_residual_blocks=6):
        super(RS_feature, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,kernel_no, kernel_size=kernel_size, stride=1, padding=kernel_size//2), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(kernel_no,kernel_size))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(kernel_no, kernel_no, kernel_size=kernel_size, stride=1, padding=kernel_size//2), nn.BatchNorm2d(kernel_no))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        return out

# =========================================================

# original SRRes, but with input kernel_no and kernel_size
class SRRes(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6,up_factor=4,
                 kernel_size=3,kernel_no=64,kauxhr=0,auxhr_chls=1):
        super(SRRes, self).__init__()
        
        # First layer
        self.conv1 = RS_feature(in_channels,kernel_no=kernel_no,kernel_size=kernel_size,n_residual_blocks=n_residual_blocks)
        
        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        # upsample_block_num = int(math.log(up_factor, 2))
        upsampling = []
        for i in range(upsample_block_num):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(kernel_no, kernel_no*factors[i]**2, 3, 1, 1),
                # nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=factors[i]),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.kaux = kauxhr
        self.aux_chls = auxhr_chls
        if self.kaux ==0: # no auxiliary field, original 
            # Final output layer
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            self.conv_aux = nn.Sequential(nn.Conv2d(kernel_no+auxhr_chls, kernel_no, 3, 1, 1),nn.PReLU())
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 

    def forward(self, x,y=None):
        out= self.conv1(x)
        out = self.upsampling(out)
        if self.kaux ==0: # no auxiliary field, original 
            out = self.conv3(out)
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            var = torch.cat((out,y),axis=1)
            out = self.conv_aux(var)
            out = self.conv3(out)
        return (out+1)/2 # tanh [-1,1] ->[0,1]

# =========================================================
#  SRRes with attention
class SRResA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6,up_factor=4,
                 kernel_size=3,kernel_no=64,kauxhr=0,auxhr_chls=1,mod_att=0,katt=0):
        super(SRResA, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,kernel_no, kernel_size=kernel_size, stride=1, padding=kernel_size//2), nn.PReLU())
        if katt>0 and mod_att>0: # mod_att is attention model type; katt: if add attenion to other conv
            self.conv1.append(AttentionBlock(kernel_no,mod_att))
        
        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlockA(kernel_no,kernel_size,mod_att))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(kernel_no, kernel_no, kernel_size=kernel_size, stride=1, padding=kernel_size//2), nn.BatchNorm2d(kernel_no))
        if katt>0 and mod_att>0: # mod_att is attention model type; katt: if add attenion to other conv
            self.conv2.append(AttentionBlock(kernel_no,mod_att))

        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        # upsample_block_num = int(math.log(up_factor, 2))
        upsampling = []
        for i in range(upsample_block_num):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(kernel_no, kernel_no*factors[i]**2, 3, 1, 1),
                # nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=factors[i]),
                nn.PReLU(),
            ]
            if katt==2 and mod_att>0: # mod_att is attention model type; katt: if add attenion to other conv
                upsampling += [AttentionBlock(kernel_no,mod_att)]
        self.upsampling = nn.Sequential(*upsampling)

        self.kaux = kauxhr
        self.aux_chls = auxhr_chls
        if self.kaux ==0: # no auxiliary field, original 
            # Final output layer
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            self.conv_aux = nn.Sequential(nn.Conv2d(kernel_no+auxhr_chls, kernel_no, 3, 1, 1),nn.PReLU())
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 

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

# =========================================================

    
    