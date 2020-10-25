from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

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
import random

# my libs
from utils import ToCuda, load_UnDP, overlay_davis, overlay_checker, overlay_color, overlay_fade, To_onehot, Dilate_scribble
from IGN_v1 import IGN

# davis
from davisinteractive.utils.scribbles import annotated_frames, scribbles2mask


class model():
    def __init__(self, frames, batch_size=1):
        self.model = IGN()
        if torch.cuda.is_available():
            print('Using GPU')
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.model.load_state_dict(torch.load('e120.pth'))
        else:
            print('Using CPU')
            self.model.load_state_dict(torch.load('e120.pth'))

        self.model.eval() # turn-off BN
        self.batch_size = batch_size
        self.init_variables(frames)
        
        
    def init_variables(self, frames):
        self.frames = frames.copy()
        self.num_objects = 1
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        # tensors
        self.Fs = torch.unsqueeze(torch.from_numpy(np.transpose(frames, (3, 0, 1, 2))).float() / 255., dim=0) # 1,3,t,h,w
        self.Es = torch.zeros(1, self.num_objects+1, self.num_frames, self.height, self.width) # inc bg 
        self.Es[:,0] = 1.0 # init bg=1, objs = 0
        # to cuda
        self.Fs, self.Es = ToCuda([self.Fs, self.Es])
        # memory
        self.keys, self.values = None, None 
        self.prev_targets = []
        # dummy run
        with torch.no_grad():
            new_key, new_value = self.model(self.Fs[:,:,0], self.Es[:,:,0], self.Es[:,:,0])
        keys = new_key.unsqueeze(2) # 1, 128, 1, H/16, W/16 
        values = new_value.unsqueeze(2)  # 1, 128, 1, H/16, W/16 
        exp_keys = keys.expand(self.batch_size,-1, -1,-1,-1) # C, 128, T, H/16, W/16 
        exp_values = values.expand(self.batch_size,-1,-1,-1,-1) # C, 128, T, H/16, W/16
        batch_Fs = self.Fs[0].transpose(0,1)
        with torch.no_grad():
            logit = self.model(batch_Fs[:self.batch_size], exp_keys, exp_values) # C, 2, H, W


    def Memorize(self, scribbles):
        # convert davis scribbles to torch
        target = scribbles['annotated_frame']
        S = scribbles2mask(scribbles, (self.height, self.width))[target]
        S = To_onehot(S, self.num_objects) # 1, no, H, W
        S = Dilate_scribble(S, self.num_objects)  # 1, no, H, W
        S = ToCuda( torch.from_numpy(S).float() )

        # compute key, value
        with torch.no_grad():
            new_key, new_value = self.model(self.Fs[:,:,target], self.Es[:,:,target], S ) 
            # 1, 128 and 512, H/16, W/16
        # update  
        if self.prev_targets == []: # init
            self.keys = new_key.unsqueeze(2) # 1, 128, 1, H/16, W/16 
            self.values = new_value.unsqueeze(2)  # 1, 128, 1, H/16, W/16 
            self.prev_targets.append(target)
            # print('[Memory] update -- add frame {}'.format(target))
        else:
            if target not in self.prev_targets:
                self.keys = torch.cat([self.keys, new_key.unsqueeze(2)], dim=2)
                self.values = torch.cat([self.values, new_value.unsqueeze(2)], dim=2)  # 1, 128, T, H/16, W/16 
                self.prev_targets.append(target)
                # print('[Memory] update -- add frame {}'.format(target))
            else: # duplicate - replace
                self.keys[:,:,self.prev_targets.index(target)] = new_key
                self.values[:,:,self.prev_targets.index(target)] = new_value
                # print('[Memory] update -- replace frame {}'.format(target))

    def Segment(self, target):
        if target == 'All':
            # print('[Query] processing frames...')
            chunks = [(x,min(x+self.batch_size, self.num_frames)) for x in range(0, self.num_frames, self.batch_size)]
            batch_Fs = self.Fs[0].transpose(0,1)
            for c in chunks:
                exp_keys = self.keys.expand(c[1]-c[0],-1, -1,-1,-1) # C, 128, T, H/16, W/16 
                exp_values = self.values.expand(c[1]-c[0],-1,-1,-1,-1) # C, 128, T, H/16, W/16
                with torch.no_grad():
                    logit = self.model(batch_Fs[c[0]:c[1]], exp_keys, exp_values) # C, 2, H, W
                    logit = logit.unsqueeze(2).transpose(0,2)
                    self.Es[:,:,c[0]:c[1]] = F.softmax(logit, dim=1)
            # print('[Query] Done.')
        else:
            with torch.no_grad():
                # frame: 1, 3, H, W  // meme: 1, 128, T, H/16, W/16
                logit = self.model(self.Fs[:,:,target], self.keys, self.values) # 1, no+1, T, H, W
                self.Es[:,:,target] = F.softmax(logit, dim=1)


    def Get_mask(self):
        return torch.round(self.Es[0,1]).data.cpu().numpy().astype(np.uint8) 

    def Get_mask_index(self, index):
        return torch.round(self.Es[0,1,index]).data.cpu().numpy().astype(np.uint8)
