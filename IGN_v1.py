from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys
 
sys.path.insert(0, '.')
from utils import *
 
print('IGN v1: Faster inference with single object // Interaction Guided Network')


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # my mask
        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # my scribble
        self.conv1_om = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # other mask (not include bg)
        self.conv1_n = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # other scribble (include bg)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_s, in_os):
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        m = torch.unsqueeze(in_m, dim=1).float()
        s = torch.unsqueeze(in_s, dim=1).float()
        # om = torch.unsqueeze(in_om, dim=1).float()
        os = torch.unsqueeze(in_os, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_p(s) + self.conv1_n(os) # + self.conv1_om(om)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        return r4, r3, r2, c1, f

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class IGN(nn.Module):
    def __init__(self):
        super(IGN, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KeyValue_M = KeyValue(1024, keydim=128, valdim=512)
        self.KeyValue_Q = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)


    def memorize(self, frame, masks, scribs): 
        # frame: 1,3,H,W / mask: 1,2,H,W / scribs: 1,2,H,W
        _, _, H, W = masks.shape # 1,2,H,W
        # pad
        (frame, masks, scribs), pad = pad_divide_by([frame, masks, scribs], 16, (frame.size()[2], frame.size()[3]))
        r4, _, _, _, _ = self.Encoder_M(frame, masks[:,1], scribs[:,1], scribs[:,0])
        k4, v4 = self.KeyValue_M(r4) # 1, 128 and 512, H/16, W/16
        return k4, v4


    def segment(self, frames, keys, values):
        '''
        support batch when num_object is 1 / 
        '''
        # meme: B, 128/512, T, H/16, W/16
        # frames: B, 3, H/16, W/16
        B, keydim, t, h, w = keys.shape # B = 1
        # pad
        [frames], pad = pad_divide_by([frames], 16, (frames.size()[2], frames.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frames)
        k4, v4 = self.KeyValue_Q(r4)   # B, dim, H/16, W/16
        
        
        m4, viz = self.Memory(keys, values, k4, v4)
        logit = self.Decoder(m4, r3, r2) # B, 2, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]
        return logit



    # def segment_frame(self, frame, keys, values, num_objects): 
    #     num_objects = num_objects[0].item()
        
    #     _, K, keydim, t, h, w = keys.shape # B = 1
    #     # pad
    #     [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

    #     r4, r3, r2, _, _ = self.Encoder_Q(frame)
    #     k4, v4 = self.KeyValue_Q(r4)   # 1, dim, H/16, W/16

    #     # expand to ---  no, c, h, w
    #     k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
    #     r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
    #     # memory select kv:(1, K, C, T, H, W)
    #     m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
    #     logits = self.Decoder(m4, r3e, r2e)
    #     ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
    #     #ps = indipendant possibility to belong to each object
        
    #     logit = self.Soft_aggregation(ps, K) # 1, K, H, W

    #     if pad[2]+pad[3] > 0:
    #         logit = logit[:,:,pad[2]:-pad[3],:]
    #     if pad[0]+pad[1] > 0:
    #         logit = logit[:,:,:,pad[0]:-pad[1]]

    #     return logit

    


    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)