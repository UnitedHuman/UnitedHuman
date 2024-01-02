import pickle
from cv2 import transform
import torch
import numpy as np
from util import patch_util, renormalize
from torchvision.utils import save_image
from util import viz_util
import torchvision.transforms as T
from torchvision import transforms

from PIL import ImageDraw, Image
import time
from PIL import Image
torch.set_grad_enabled(False)
import random
import os
import glob
import argparse 
from PIL import ImageDraw 
import cv2

from util.patch_util import construct_transformation_matrix
    
def comb_image(imgs, axis=1):
    concatenated = Image.fromarray(
        np.concatenate(
            [np.array(x) for x in imgs],
            axis=axis
        )
    )
    return concatenated


def main(args):
    path = args.path_list
    only_mean = args.only_mean #False
    model_name = path.split('/')[-1].replace('.pkl','')

    output_path = 'output/'+model_name+'_inference'
    os.makedirs(output_path, exist_ok=True)

    
    try:
        with open(path, 'rb') as f:
            G_base = pickle.load(f)['G_ema'].cuda() 
    except:
        f = open(path,'rb')
        G_base = torch.load(f,map_location='cpu')['G_ema'].cuda()
            

    if only_mean:
        seeds = [0]
    else:
        seeds = range(5)
    identity = torch.eye(3).unsqueeze(0).cuda()
    full_size_list = [args.target_size] #[2048]#[256,512,1024,2048]
    for seed in seeds:
        if not only_mean:
            rng = np.random.RandomState(seed)
            z = torch.from_numpy(rng.standard_normal(512)).float()
            z = z[None].cuda()
            c = None
            ws = G_base.mapping(z, c, truncation_psi=0.8)

        z_samples = np.random.RandomState(123).randn(10000, 512)
        w_samples = G_base.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
        
        if not only_mean:
            pil_imgs = []
            for full_size in full_size_list:
                full = torch.zeros([1, 3, full_size, full_size])
                patches = patch_util.generate_full_from_patches(full_size, G_base.img_resolution)
                for bbox, transform in patches:
                    img = patch_util.scale_condition_wrapper(G_base, ws, transform[None].cuda(), noise_mode='const', force_fp32=True)
                    full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
                full[:, :, :, 0:full_size//4] = 1
                full[:, :, :, 3*full_size//4:] = 1
                pil_img = renormalize.as_image(full[0]) 
                pil_imgs.append(pil_img)
            combined = comb_image(pil_imgs)

        ws_mean = torch.mean(w_samples[:, :, :], dim=0, keepdim=True) # [N, L, C]
        if only_mean:
            ws = ws_mean
            seed = 'meanlatent'
        else: 
            ori_ws = ws.clone()
            ws[:, :8, :] = ws_mean[:, :8, :]

        pil_imgs = []
        for full_size in full_size_list:
            full = torch.zeros([1, 3, full_size, full_size])
            patches = patch_util.generate_full_from_patches(full_size, G_base.img_resolution)
            for bbox, transform in patches:
                img = patch_util.scale_condition_wrapper(G_base, ws, transform[None].cuda(), noise_mode='const', force_fp32=True)
                full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
            full[:, :, :, 0:full_size//4] = 1
            full[:, :, :, 3*full_size//4:] = 1
            pil_img = renormalize.as_image(full[0]) 
            pil_imgs.append(pil_img)
        combined2 = comb_image(pil_imgs)

        if only_mean:
            combined2.save(os.path.join(output_path, f"{seed}_combined.png")) 
        else:
            comb_all = comb_image([combined, combined2], axis=0)
            comb_all.save(os.path.join(output_path, f"{seed}_combined.png"))
        

if __name__ == '__main__':
    t1 = time.time()
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_formatter)
    parser.add_argument('--path_list', type=str)
    parser.add_argument('--target_size', type=str, default=1024)
    parser.add_argument('--only_mean', action='store_true')
    print('parsing arguments')
    cmd_args = parser.parse_args()
    main(cmd_args)

    print('total time elapsed: ', str(time.time() - t1))
