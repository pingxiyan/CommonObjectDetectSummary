# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:12:25 2019

@author: Sandy
"""

import torch
from network import Net

def cvt_model(pickle_model, script_model):
    print("start convert")
    print("Initiate model ...")
    model = Darknet(opt.model_def).to(device)
    
    if pickle_model.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(pickle_model)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(pickle_model))
    model.eval()

    checkpoint = torch.load(pickle_model)
    
    #example = torch.rand(1,3,32,32).cuda() 
    example = torch.rand(1,3,32,32).cpu()
    
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(script_model)
    
    print("convert complete")
    
if __name__ == '__main__':
    #pickle_model = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2663.pt"
    #script_model = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2831.pts"
    
    pickle_model = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.2715.pt"
    script_model = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.2715.pts"
    cvt_model(pickle_model, script_model)