# -*- coding: utf-8 -*-
"""
Data_loader.py is to load the splited images into samples in list type, each element in this list is a dictionary unit named sample
Each sample consists of 1 target image and two reference images

"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
import random
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pdb
import custom_transforms
import main
from params import args

sequence_length = 3 # Input sequece are 3 images
demi_length = (sequence_length-1)//2

def get_lists(ds_path):
    # ds = './dataset'
    ll = os.listdir(ds_path)
    ll = [ds_path + '/' + x for x in ll]
    # ll = ['./dataset/video1', './dataset/video2', ...]

    
    seq_list = []
    
    for x in ll:
        imgs = os.listdir(x)
        imgs = [x + '/' + img for img in imgs]
         
        seq_list.append(imgs)
       
    return seq_list  # List[ List[img_path] ]


def get_shifts(sequence_length):
    # 3 -> [-2, -1]
    # 5 -> [-2, -1, 1, 2]
    assert sequence_length%2 == 1 and sequence_length > 1, \
    'sequence_length must be odd and > 1'
    
    shifts = list(range(-demi_length-1, -demi_length+1))
    #pdb.set_trace() 
    #shifts.pop(demi_length)
    
    # print(shifts)
    return shifts   


def get_samples(seq_list, sequence_length,AoI):
    shifts = get_shifts(sequence_length)
    samples = []
    
    print("AoI is %d"%AoI)
    for imgs in seq_list:
        assert len(imgs) > sequence_length, 'dataset is too small!'
        for i in range(demi_length+1, len(imgs)-AoI): # 1，321-1
            #pdb.set_trace() 
            sample = {'tgt': imgs[i+AoI], 
                      'ref_imgs': []
                      }
            
            for j in shifts: # shifts = [-2, -1]
                sample['ref_imgs'].append(imgs[i+j])  # tgt is i   ref is i-1 and i-2
                
            samples.append(sample) 
        random.shuffle(samples)
      
    return samples



def get_multi_scale_intrinsics(intrinsics, num_scales):
    """Returns multiple intrinsic matrices for different scales."""

    intrinsics_multi_scale = np.zeros( (4,3,3), dtype = np.float32)
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        intrinsics_copy = intrinsics.copy()
        
        intrinsics_copy[0] = intrinsics_copy[0]*( 1/ (2**s) )
        intrinsics_copy[1] = intrinsics_copy[1]*( 1/ (2**s) )
        
        intrinsics_multi_scale[s, :,:] = intrinsics_copy
    
    return intrinsics_multi_scale  # [4, 3, 3]


def get_multi_scale_inv_intrinsics(intrinsics, num_scales):
    inv_intrinsics_multi_scale = np.zeros( (4,3,3), dtype = np.float32)
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        intrinsics_copy = intrinsics.copy()
        
        intrinsics_copy[0] = intrinsics_copy[0]*( 1/ (2**s) )
        intrinsics_copy[1] = intrinsics_copy[1]*( 1/ (2**s) )
        
        inv_intrinsics_multi_scale[s, :,:] = np.linalg.inv(intrinsics_copy) 
        # 求了相机逆矩阵
    
    return inv_intrinsics_multi_scale



def load_as_float(path):
    return imread(path).astype(np.float32)

class SequenceFolder():
    def init(self, 
                 ds_path, 
                 AoI,
                 seed=0, 
                 sequence_length = 3,
                 num_scales = 4):
        np.random.seed(seed)
        random.seed(seed)

        self.num_scales = num_scales
        
        seq_list = get_lists(ds_path)
        self.samples = get_samples(seq_list, sequence_length,AoI)
        
        # get by calib.py
        self.intrinsics = np.array([
                                 [1.14183754e+03, 0.00000000e+00, 6.28283670e+02],
                                 [0.00000000e+00, 1.13869492e+03, 3.56277189e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 ]).astype(np.float32)
        
        # The original size of the picture taken by my phone camera is 1280 x 720.
        # if your original picture size is not 1280 x 720, change the two numbers below
        # resize 1280 x 720 -> 416 x 128
        self.intrinsics[0] = self.intrinsics[0]*(416.0/1280.0)
        self.intrinsics[1] = self.intrinsics[1]*(128.0/720.0)
        
        self.ms_k     = get_multi_scale_intrinsics(self.intrinsics, self.num_scales)
        self.ms_inv_k = get_multi_scale_inv_intrinsics(self.intrinsics, self.num_scales)
        
        ######################
        self.to_tensor = custom_transforms.Compose([ custom_transforms.ArrayToTensor() ])
        self.to_tensor_norm = custom_transforms.Compose([ custom_transforms.ArrayToTensor(),
                                                     custom_transforms.Normalize(
                                                             mean=[0.485, 0.456, 0.406],
                                                             std =[0.229, 0.224, 0.225])
                                                  ])
        
    def __getitem__(self, index):
        sample = self.samples[index]
        # np.copy(sample['intrinsics'])
        
        tgt_img = load_as_float(sample['tgt'])   # Read the target image
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        ref_imgs.insert(demi_length+1, tgt_img) #tgt is prediction image with an AoI
        image_stack_origin = ref_imgs  

        
        
        image_stack = self.to_tensor( image_stack_origin.copy() )
        image_stack_norm = self.to_tensor_norm( image_stack_origin.copy() )
        
        intrinsic_mat = torch.from_numpy(self.ms_k)
        intrinsic_mat_inv = torch.from_numpy(self.ms_inv_k)
        
        return image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv
       

    def __len__(self):
        return len(self.samples)




                                                  
                                                     
        
if __name__ == '__main__':
    ds_path = './dataset'   
    seqlen = 3
    a = [None]*(seqlen-1)
    demi_length = (seqlen-1)//2
    a.insert(demi_length, 6)
    
    print(a)
    