import numpy as np
import random
import torch
import math
from math import exp
from PIL import Image,ImageDraw
import random
import cv2
import os
import math
from metrics import equivariance
from util.image_util import uncrop

###################
def apply_affine_batch(img, transform):
    # hacky .. apply affine transformation with cuda kernel in batch form
    crops = []
    masks = []
    for i, t in zip(img, transform):
        crop, mask = equivariance.apply_affine_transformation(
            i[None], t.inverse())
        crops.append(crop)
        masks.append(mask)
    crops = torch.cat(crops, dim=0)
    masks = torch.cat(masks, dim=0)
    return crops, masks


def construct_transformation_matrix(limits):
    # limits is a list of [(y_min, y_max), (x_min, x_max)]
    # in normalized coordinates from -1 to 1
    x_limits = limits[1]
    y_limits = limits[0]
    theta = torch.zeros(2, 3)
    tx = np.sum(x_limits) / 2
    ty = np.sum(y_limits) / 2
    s = x_limits[1] - tx
    assert(np.abs((x_limits[1] - tx) - (y_limits[1] - ty)) < 1e-9)
    theta[0, 0] = s
    theta[1, 1] = s
    theta[0, 2] = (tx) / 2
    theta[1, 2] = (ty) / 2
    transform = torch.zeros(3, 3)
    transform[:2, :] = theta
    transform[2, 2] = 1.0
    return transform
    


def generate_by_keypoints(query_idx, crop_size = 256, scale=6, apose_source='015'):
    if apose_source == '015':
        apose_2048 = np.array([1026,  260, 1024,  456,  868,  456,  823,  734,  812,  990, 
                                1179, 456, 1213,  723, 1224,  990, 1024,  957,  912,  957, 
                                912, 1380, 912, 1792, 1124,  957, 1124, 1380, 1113, 1803,  
                                992,  223, 1059, 223,  954,  245, 1097,  243, 1146, 1914, 
                                1179, 1892, 1079, 1825, 890, 1914,  857, 1892,  946, 1825])  # apose from exp015
    apose = apose_2048.reshape([25, 2]) / (2048//crop_size)# 8
    a_x, a_y = apose[query_idx] * scale
    a_x =  a_x - crop_size // 2 
    a_y =  a_y - crop_size // 2 
    new_ssize = crop_size * scale
    limits = [
            (a_y/(new_ssize)*2-1, (a_y+crop_size) /(new_ssize)*2-1),
            (a_x/(new_ssize)*2-1, (a_x+crop_size) /(new_ssize)*2-1)
    ]
    transform = construct_transformation_matrix(limits)
    return transform

def generate_full_from_patches(new_size, patch_size=256):
    # returns the bounding boxes and transformations needed to 
    # piece together patches of size patch_size into a 
    # full image of size new_size
    patch_params = []
    for y in range(0, new_size, patch_size):
        for x in range(0, new_size, patch_size):
            if y + patch_size > new_size:
                y = new_size - patch_size
            if x + patch_size > new_size:
                x = new_size - patch_size
            limits = [(y/(new_size)*2-1, (y+patch_size) /(new_size)*2-1),
              (x/(new_size)*2-1, (x+patch_size) /(new_size)*2-1)]
            transform = construct_transformation_matrix(limits)
            patch_params.append(((y, y+patch_size, x, x+patch_size), transform))
    return patch_params



def compute_scale_inputs(G, w, transform):
    if transform is None:
        scale = torch.ones(w.shape[0], 1).to(w.device)
    else:
        scale = 1/transform[:, [0], 0]
    scale = G.scale_norm(scale)
    mapped_scale = G.scale_mapping(scale, None)
    return scale, mapped_scale

def scale_condition_wrapper(G, w, transform, **kwargs):
    # convert transformation matrix into scale input
    # and pass through scale mapping network
    if not G.scale_mapping_kwargs:
        img = G.synthesis(w, transform=transform, **kwargs)
        return img
    scale, mapped_scale = compute_scale_inputs(G, w, transform)
    img = G.synthesis(w, mapped_scale=mapped_scale, transform=transform, **kwargs)
    return img


import torch.nn.functional as F
def generate_full_img(G, ws, full_size, scale, patch_size=256):
    full = torch.zeros([ws.shape[0], 3, full_size, full_size])
    resize_to = patch_size // scale
    patches = generate_full_from_patches(full_size, G.img_resolution)
    for bbox, transform in patches:
        img = scale_condition_wrapper(G, ws, transform[None].cuda(), noise_mode='const', force_fp32=True)
        full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
    full = F.interpolate(full, (patch_size, patch_size), mode='area')
    return full