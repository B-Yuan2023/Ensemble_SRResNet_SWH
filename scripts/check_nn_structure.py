#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:16:52 2024
check structure of the model 
@author: g260218
"""

import os
import torch
from models import GeneratorResNet
from mod_srcnn import SRCNN,FSRCNN,SRCNN1,FSRCNN1
from mod_srres import SRRes,SRResA
from mod_srres_ms import SRRes_MS,SRRes_MS1,SRRes_MS2

if __name__ == '__main__':

    nchl_i,nchl_o = 1,1
    nchl = nchl_o
    
    up_factor=16
    hr_shape = (128, 128)
    lr_shape = (int(hr_shape[0]/up_factor), int(hr_shape[1]/up_factor))

    knn = 6
    ker_size = [3,5]
    kernel_no = 64
    residual_blocks= 6
    dsm = [128,32,4]  # default [56,12,4]
    
    mod_att = 0
    katt = 0
    
    # Initialize generator     
    if knn ==0:
        generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                    n_residual_blocks=residual_blocks,up_factor=up_factor)
    elif knn == 1:
        generator = SRRes(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=residual_blocks,
                                    up_factor=up_factor,kernel_size=ker_size,kernel_no=kernel_no)
    elif knn == 2:
        generator = SRResA(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=residual_blocks,
                                    up_factor=up_factor,kernel_size=ker_size,kernel_no=kernel_no,
                                    mod_att=mod_att,katt=katt)
    elif knn == 4:
        generator = SRRes_MS(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor,kernel_size=ker_size)
    elif knn == 5:
        generator = SRRes_MS1(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor,kernel_size=ker_size)
    elif knn == 6:
        generator = SRRes_MS2(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor,kernel_size=ker_size)
    elif knn == 7:
        generator = FSRCNN(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor,d=dsm[0],s=dsm[1],m=dsm[2])
    elif knn == 8:
        generator = FSRCNN1(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor,d=dsm[0],s=dsm[1],m=dsm[2])
    elif knn == 9:
        generator = SRCNN(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor)
    elif knn == 10:
        generator = SRCNN1(in_channels=nchl_i, out_channels=nchl_o,up_factor=up_factor)
                
    
    num_params = sum(p.numel() for p in generator.parameters())
    print(generator) # model infomation of each layer, without no. parameters. 
    print(f"Total number of learnable parameters: {num_params}")
    
    # batch_size = 12 
    # dat_lr = torch.randn(batch_size, nchl_i, lr_shape[0], lr_shape[1]) 
    # gen_hr = generator(dat_lr)
    
    # from torchsummary import summary
    # summary(generator, dat_lr.shape)
    
    # from torchviz import make_dot
    # Save the graph to a file (optional)
    # dot = make_dot(gen_hr, params=dict(generator.named_parameters()))
    # out_path = 'nn_structure/'
    # os.makedirs(out_path, exist_ok=True)
    # ofname = "nn_b%d_s%d_batch%d_nch%d_lr%d_%d" % (
    #     residual_blocks,up_factor,batch_size,nchl_i,lr_shape[0],lr_shape[1]) + '.png'
    # dot.render(out_path+ofname, format='png')

    # # Display the graph
    # dot.view()
    
    # from torchview import draw_graph
    # model_graph = draw_graph(generator(), input_size=(1,3,224,224), expand_nested=True)
    # model_graph.visual_graph
    