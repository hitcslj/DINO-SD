from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import time
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import crop
import pdb

from augmentation.augmix import augmix
from augmentation.config import STD,MEAN
import torch.nn.functional as F

mean = torch.tensor(MEAN).view(-1, 1, 1)
std = torch.tensor(STD).view(-1, 1, 1)

def pil_loader(path, k=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            if k!=1:
                width, height = img.size
                new_width = int(width / k)
                new_height = int(height / k)
                img = img.resize((new_width, new_height), Image.BILINEAR)
            return img
        
class MonoDataset(data.Dataset):
    def __init__(self,
                 opt,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super(MonoDataset, self).__init__()
        self.opt = opt
        self.height = height
        self.width  = width
        self.crop_height = self.opt.height_crop 
        self.crop_width = self.opt.width_crop
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.resize = {}
        self.crop = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
            
    
    def __len__(self):
        return len(self.filenames)
    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # dw, dh = torch.randint(self.opt.width_ori - self.crop_width), torch.randint(self.opt.height - self.crop_height)
        # import pdb
        # pdb.set_trace()
        dw, dh = random.randint(0, self.opt.width_ori - self.crop_width), random.randint(0, self.opt.height_ori - self.crop_height)
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                if self.is_train and not self.opt.no_crop and i == -1:
                    for index_spatial in range(6):
                        inputs[(n, im, i)][index_spatial] = crop(inputs[(n, im, i)][index_spatial], dw, dh, self.crop_height, self.crop_width)

        
        for k in list(inputs):
            if "color" in k:
                n, im, i = k

                for i in range(self.num_scales):
                    inputs[(n, im, i)] = []
                    if im==0:
                        inputs[(n + "_aug_1", im, i)] = []
                        inputs[(n + "_aug_2", im, i)] = []
                    for index_spatial in range(6):
                        inputs[(n, im, i)].append(self.resize[i](inputs[(n, im, i - 1)][index_spatial]))

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                for index_spatial in range(6):
                    pic = f[index_spatial]
                    aug_tensor=self.to_tensor(pic)
                    aug_tensor = (aug_tensor-mean)/std
                    inputs[(n, im, i)][index_spatial] = self.to_tensor(pic)
                    if im == 0 and i != -1:
                        aug1 = color_aug(pic)
                        aug2 = color_aug(pic)
                        inputs[(n + "_aug_1", im, i)].append(self.to_tensor(aug1))
                        inputs[(n + "_aug_2", im, i)].append(self.to_tensor(aug2))
                inputs[(n, im, i)] = torch.stack(inputs[(n, im, i)], dim=0)       
                if im == 0 and i != -1:
                    inputs[(n + "_aug_1", im, i)] = torch.stack(inputs[(n + "_aug_1", im, i)], dim=0)
                    inputs[(n + "_aug_2", im, i)] = torch.stack(inputs[(n + "_aug_2", im, i)], dim=0)

        inputs[("gt")] = torch.stack(inputs[("gt")], dim=0)
        inputs[("gt")] = crop(inputs[("gt")], dw, dh, self.crop_height, self.crop_width) if not self.opt.no_crop else inputs[("gt")]
        
    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train
        do_flip = self.is_train and (not self.opt.use_sfm_spatial) and (not self.opt.joint_pose) and random.random() > 0.5
        frame_index = self.filenames[index].strip().split()[0]
        self.get_info(inputs, frame_index, do_flip)
        if not self.is_train:
            self.frame_idxs = [0]
            
        for scale in range(self.num_scales):
            for frame_id in  self.frame_idxs:
                inputs[("K", frame_id, scale)] = []
                inputs[("inv_K", frame_id, scale)] = []
    
        for index_spatial in range(6):
            for scale in range(self.num_scales):
                for frame_id in  self.frame_idxs:
                    K = inputs[('K_ori', frame_id)][index_spatial].copy()

                    if self.is_train and not self.opt.no_crop:
                        resize_w = self.crop_width / inputs['width_ori'][index_spatial]
                        resize_h = self.crop_height / inputs['height_ori'][index_spatial]

                        K[0, 2] = K[0, 2] * resize_w
                        K[1, 2] = K[1, 2] * resize_h


                    K[0, :] *= (self.width // (2 ** scale)) / self.crop_width if self.is_train and not self.opt.no_crop else inputs['width_ori'][index_spatial]
                    K[1, :] *= (self.height // (2 ** scale)) / self.crop_height if self.is_train and not self.opt.no_crop else inputs['height_ori'][index_spatial]

                    inv_K = np.linalg.pinv(K)        
                    inputs[("K", frame_id, scale)].append(torch.from_numpy(K))
                    inputs[("inv_K", frame_id, scale)].append(torch.from_numpy(inv_K))            
        for scale in range(self.num_scales):
            for frame_id in  self.frame_idxs:
                inputs[("K",frame_id, scale)] = torch.stack(inputs[("K",frame_id, scale)], dim=0)
                inputs[("inv_K",frame_id, scale)] = torch.stack(inputs[("inv_K", frame_id,scale)], dim=0)
        
        if do_color_aug:
            color_aug = augmix
        else:
            color_aug = lambda x: x
        
        self.preprocess(inputs, color_aug)

        del inputs[("color", 0, -1)]
        if self.is_train:
            for i in self.frame_idxs[1:]:
                del inputs[("color", i, -1)]
            for i in self.frame_idxs:
                del inputs[('K_ori', i)]
        else:
            del inputs[('K_ori', 0)]
            
        del inputs['width_ori']
        del inputs['height_ori']

        
        if 'depth' in inputs.keys():
            inputs['depth'] = torch.from_numpy(inputs['depth'])
        
        if 'raw_image' in inputs.keys():
            inputs['raw_image'] = torch.from_numpy(inputs['raw_image'])

        if self.is_train:
            inputs["pose_spatial"] = torch.from_numpy(inputs["pose_spatial"])
            for i in self.frame_idxs[1:]:
                inputs[("pose_spatial", i)] = torch.from_numpy(inputs[("pose_spatial", i)])
                
            if self.opt.use_sfm_spatial:
                for j in range(len(inputs['match_spatial'])):
                    inputs['match_spatial'][j] = torch.from_numpy(inputs['match_spatial'][j])
            
            if self.opt.use_fix_mask:
                inputs["mask"] = []
                for i in range(6):
                    temp = cv2.resize(inputs["mask_ori"][i], (self.width, self.height))
                    temp = temp[..., 0]
                    temp = (temp == 0).astype(np.float32)
                    inputs["mask"].append(temp)
                inputs["mask"] = np.stack(inputs["mask"], axis=0)
                inputs["mask"] = np.tile(inputs["mask"][:, None], (1, 2, 1, 1))
                inputs["mask"] = torch.from_numpy(inputs["mask"])
                if do_flip:
                    inputs["mask"] = torch.flip(inputs["mask"], [3])
                del inputs["mask_ori"]

        return inputs
    
    def get_info(self, inputs, index, do_flip):
        raise NotImplementedError