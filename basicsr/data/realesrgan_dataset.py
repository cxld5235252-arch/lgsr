import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path

import albumentations

import torch.nn.functional as F
from torch.utils import data as data

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, mode='training'):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # file client (lmdb io backend)
        
        self.gt_root = opt['gt_root']
        self.tele_root = opt['tele_root']
        self.lq_root = opt['lq_root']
        self.paths = sorted(os.listdir(self.gt_root))
        self.tele_paths = sorted(os.listdir(self.tele_root))
        self.lq_paths = sorted(os.listdir(self.lq_root))

        self.mode = mode

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        # assert self.paths[index] == self.tele_paths[index]
        gt_path = os.path.join(self.gt_root, self.paths[index])
        lq_path = os.path.join(self.lq_root, self.lq_paths[index])
        tele_path = os.path.join(self.tele_root, self.tele_paths[index])

        img_gt = cv2.imread(gt_path) 
        img_lq = cv2.imread(lq_path) 
        img_tele = cv2.imread(tele_path) 

        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)/255.
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)/255.
        img_tele = cv2.cvtColor(img_tele, cv2.COLOR_BGR2RGB)/255.

        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=self.opt['gt_size']),
                    albumentations.CenterCrop(self.opt['gt_size'], self.opt['gt_size']),
                    ])
            img_gt = self.test_aug(image=img_gt)['image']
        elif self.mode == 'training':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        if self.mode == 'training':
            # -------------------- Do augmentation for training: flip, rotation -------------------- #
            h, w = img_gt.shape[0:2]
            img_lq = cv2.resize(img_lq, (w, h))
            
            img_gt, img_lq, img_tele = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'], img_tele,img_lq)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        h, w = img_gt.shape[0:2]
    
        def add_gaussian_noise(tensor, mean=0.0, std=0.02):
            noise = torch.randn(tensor.size()) * std + mean
            noisy_tensor = tensor + noise
            # 确保张量仍然在 [0, 1] 范围内
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
            return noisy_tensor

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        img_gt = img_gt - imagenet_mean
        img_gt = img_gt / imagenet_std
        img_tele = img_tele - imagenet_mean
        img_tele = img_tele / imagenet_std
        img_lq = img_lq - imagenet_mean
        img_lq = img_lq / imagenet_std
        
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0]
        img_tele = img2tensor([img_tele], bgr2rgb=False, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=False, float32=True)[0]
        # wide加高斯噪声
        # mean = random.uniform(0.0, 0.1)
        # img_lq = add_gaussian_noise(img_lq, mean)
        # print(img_gt.mean(),',',img_tele.mean())
        return_d = {'gt': img_gt, 'tele' : img_tele, 'lq':img_lq, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)

 