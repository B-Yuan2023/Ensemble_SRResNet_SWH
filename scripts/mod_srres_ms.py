#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:12:55 2024

@author: g260218
"""

import torch.nn as nn
import torch
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

# multi-kernel size before the residual block
class SRRes_MS(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6,up_factor=4,
                 kernel_size=3,kernel_no=64,kauxhr=0,auxhr_chls=1):
        super(SRRes_MS, self).__init__()

        if isinstance(kernel_size, (int, float)): 
            kernel_size = [kernel_size,]  # to make kernel_size a list
        
        nks = len(kernel_size)
        # First layer
        self.conv1 = nn.ModuleList()
        for i in range(nks):
            iks = kernel_size[i]
            self.conv1.append(nn.Sequential(nn.Conv2d(in_channels,kernel_no, kernel_size=iks, stride=1, padding=iks//2), nn.PReLU()))
        # input is concatenated conv1
        self.mslayer = nn.Sequential(
            nn.Conv2d(kernel_no*nks,kernel_no, kernel_size=7, stride=1, padding=7//2), nn.PReLU(),
            nn.Conv2d(kernel_no,kernel_no, kernel_size=5, stride=1, padding=5//2), nn.PReLU(),
            nn.Conv2d(kernel_no,kernel_no, kernel_size=3, stride=1, padding=3//2), nn.PReLU(),
            )
        
        # Residual blocks            
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(kernel_no))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(kernel_no, kernel_no, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(kernel_no))

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
        ms_out0 = []
        for layer in self.conv1:
            ms_out0.append(layer(x))
        out0 = torch.cat(ms_out0,1)
        out1= self.mslayer(out0)
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



# multi-kernel size before the upsampling block
class SRRes_MS1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6,up_factor=4,
                 kernel_size=3,kernel_no=64,kauxhr=0,auxhr_chls=1):
        super(SRRes_MS1, self).__init__()

        if isinstance(kernel_size, (int, float)): 
            kernel_size = [kernel_size,]  # to make kernel_size a list
        
        nks = len(kernel_size)
        # First layer
        self.conv1 = nn.ModuleList()
        for i in range(nks):
            iks = kernel_size[i]
            self.conv1.append(RS_feature(in_channels,kernel_no=kernel_no,kernel_size=iks,n_residual_blocks=n_residual_blocks))
        # input is concatenated conv1
        self.mslayer = nn.Sequential(
            nn.Conv2d(kernel_no*nks,kernel_no, kernel_size=7, stride=1, padding=7//2), nn.PReLU(),
            nn.Conv2d(kernel_no,kernel_no, kernel_size=5, stride=1, padding=5//2), nn.PReLU(),
            nn.Conv2d(kernel_no,kernel_no, kernel_size=3, stride=1, padding=3//2), nn.PReLU(),
            )
        
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
        ms_out0 = []
        for layer in self.conv1:
            ms_out0.append(layer(x))
        out0 = torch.cat(ms_out0,1)
        out= self.mslayer(out0)
        out = self.upsampling(out)
        if self.kaux ==0: # no auxiliary field, original 
            out = self.conv3(out)
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            var = torch.cat((out,y),axis=1)
            out = self.conv_aux(var)
            out = self.conv3(out)
        return (out+1)/2 # tanh [-1,1] ->[0,1]

# =========================================================


# multi-kernel size within upsampling block

# upsampling module with specified kernel_size
class upsample_mod(nn.Module):
    def __init__(self,kernel_no=64,kernel_size=3,up_factor=4):
        # input kernel_size should be a number 
        super(upsample_mod, self).__init__()
        # Upsampling layers
        factors = prime_factors(up_factor)
        upsample_block_num = len(factors)
        # upsample_block_num = int(math.log(up_factor, 2))
        upsampling = []
        for i in range(upsample_block_num):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(kernel_no, kernel_no*factors[i]**2, kernel_size, 1, kernel_size//2),
                nn.PixelShuffle(upscale_factor=factors[i]),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
    def forward(self, x):
        out = self.upsampling(x)
        return out

class SRRes_MS2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6,up_factor=4,
                 kernel_size=3,kernel_no=64,kauxhr=0,auxhr_chls=1):
        super(SRRes_MS2, self).__init__()

        if isinstance(kernel_size, (int, float)): 
            kernel_size = [kernel_size,]  # to make kernel_size a list
        
        nks = len(kernel_size)
        # feature extraction using residual modules with varies kerenel_size
        # self.conv1 = nn.ModuleList()
        # for i in range(nks):
        #     iks = kernel_size[i]
        #     self.conv1.append(RS_feature(in_channels,kernel_no,iks,n_residual_blocks))
        self.conv1 = RS_feature(in_channels,kernel_no,3,n_residual_blocks)
        
        # transition between res and upsampling modules, input is concatenated conv1
        # self.conv2 = nn.ModuleList()
        # for i in range(nks):
        #     iks = kernel_size[i]
        #     self.conv2.append(nn.Sequential(nn.Conv2d(kernel_no*nks,kernel_no, iks, 1, iks//2), nn.PReLU))
        # self.conv2 = nn.Sequential(nn.Conv2d(kernel_no*nks,kernel_no, 3, 1, 3//2), nn.PReLU())
        
        # upsampling module with multiple kernal sizes
        self.upsampling_ms = nn.ModuleList()
        for i in range(nks):
            iks = kernel_size[i]
            self.upsampling_ms.append(upsample_mod(kernel_no,iks,up_factor))

        # use concatenated data from upsampling with multiple kernel sizes
        self.kaux = kauxhr
        self.aux_chls = auxhr_chls
        if self.kaux ==0: # no auxiliary field, original 
            # Final output layer
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no*nks, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            self.conv_aux = nn.Sequential(nn.Conv2d(kernel_no*nks+auxhr_chls, kernel_no, 3, 1, 1),nn.PReLU())
            self.conv3 = nn.Sequential(nn.Conv2d(kernel_no*nks, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()) 

    def forward(self, x,y=None):
        out = self.conv1(x)
        ms_out0 = []
        for layer in self.upsampling_ms:
            ms_out0.append(layer(out))
        out = torch.cat(ms_out0,1)
        if self.kaux ==0: # no auxiliary field, original 
            out = self.conv3(out)
        elif self.kaux==1:  # add auxiliary after upsampling, input channel is 64
            var = torch.cat((out,y),axis=1)
            out = self.conv_aux(var)
            out = self.conv3(out)
        return (out+1)/2 # tanh [-1,1] ->[0,1]    