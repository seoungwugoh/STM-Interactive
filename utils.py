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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import copy
import cv2
import random
import glob
from scipy.ndimage.morphology import binary_erosion, binary_dilation

def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs

def To_onehot(mask, num_objects):
    S = np.zeros((1, num_objects+1, mask.shape[0], mask.shape[1]))
    for o in range(num_objects+1):
        S[0,o] = (mask == o).astype(np.float32)
    return S
            
def Dilate_scribble(mask, num_objects):
    new_mask = np.zeros_like(mask)
    for o in range(num_objects+1): # include bg scribbles
        bmask = (mask[0,o] > 0.5).astype(np.uint8)
        new_mask[0,o] = cv2.dilate(bmask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=2)
    return new_mask


def load_frames(path, size=None, num_frames=None):
    fnames = glob.glob(os.path.join(path, '*.jpg')) 
    fnames.sort()
    frame_list = []
    for i, fname in enumerate(fnames):
        if size:
            frame_list.append(np.array(Image.open(fname).convert('RGB').resize((size[0], size[1]), Image.BICUBIC), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))
        if num_frames and i > num_frames:
            break
    frames = np.stack(frame_list, axis=0)
    return frames

def load_masks(path, size=None, num_frames=None):
    fnames = glob.glob(os.path.join(path, '*.jpg')) 
    fnames.sort()
    mask_list = []
    for i, fname in enumerate(fnames):
        mname = fname[:-4] + '.png'
        mname = mname.replace('sequences/', 'masks/')
        if size:
            mask_list.append( (np.array(Image.open(mname).convert('P').resize((size[0], size[1]), Image.NEAREST), dtype=np.uint8) > 0.5).astype(np.uint8) )
        else:
            mask_list.append( (np.array(Image.open(mname).convert('P'), dtype=np.uint8) > 0.5).astype(np.uint8) )
        if num_frames and i > num_frames:
            break
    masks = np.stack(mask_list, axis=0)
    return masks

def load_frames_and_masks(path, size=None, num_frames=None):
    frames = load_frames(path, size=size, num_frames=num_frames)
    try:
        masks = load_masks(path, size=size, num_frames=num_frames)
    except: 
        masks = None
    return frames, masks


def load_UnDP(path):
    # load dataparallel wrapped model properly
    state_dict = torch.load(path, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def overlay_cont(image,mask,rgb=[255,0,255],cscale=2,alpha=0.5):
    im_overlay = image.copy()
    binary_mask = mask == 1
    #image[binary_mask] = image[binary_mask]*alpha + (1-alpha) * np.array(rgb, dtype=np.uint8)[None, None, :]
    countours = binary_dilation(binary_mask, iterations=1) ^ binary_erosion(binary_mask, iterations=1)
    im_overlay[countours,:] = rgb
    return im_overlay.astype(image.dtype)


def overlay_davis(image,mask,rgb=[255,0,0],cscale=2,alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()

    foreground = im_overlay*alpha + np.ones(im_overlay.shape)*(1-alpha) * np.array(rgb, dtype=np.uint8)[None, None, :]
    binary_mask = mask == 1
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,:] = 0
    return im_overlay.astype(image.dtype)


def checkerboard(img_size, block_size):
    width = int(np.maximum( np.ceil(img_size[0] / block_size), np.ceil(img_size[1] / block_size)))
    b = np.zeros((block_size, block_size), dtype=np.uint8) + 32
    w = np.zeros((block_size, block_size), dtype=np.uint8) + 255 - 32
    row1 = np.hstack([w,b]*width)
    row2 = np.hstack([b,w]*width)
    board = np.vstack([row1,row2]*width)
    board = np.stack([board, board, board], axis=2)
    return board[:img_size[0], :img_size[1], :] 

BIG_BOARD = checkerboard([1000, 1000], 20)
def overlay_checker(image,mask):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    # board = checkerboard(image.shape[:2], block_size=20)
    board = BIG_BOARD[:im_overlay.shape[0], :im_overlay.shape[1], :].copy()
    binary_mask = (mask == 1)
    # Compose image
    board[binary_mask] = im_overlay[binary_mask]
    return board.astype(image.dtype)

def overlay_color(image,mask, rgb=[255,0,255]):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    board = np.ones(image.shape, dtype=np.uint8) * np.array(rgb, dtype=np.uint8)[None, None, :]
    binary_mask = (mask == 1)
    # Compose image
    board[binary_mask] = im_overlay[binary_mask]
    return board.astype(image.dtype)


def overlay_fade(image, mask):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    im_overlay = image.copy()

    # Overlay color on  binary mask
    binary_mask = mask == 1
    not_mask = mask != 1

    # Compose image
    im_overlay[not_mask] = 0.6 * im_overlay[not_mask]


    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,0] = 0
    im_overlay[countours,1] = 255
    im_overlay[countours,2] = 255

    return im_overlay.astype(image.dtype)
