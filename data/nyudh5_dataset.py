import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging

import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py

logger = setup_logging(__name__)


IMG_EXTENSIONS = ['.h5',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

class NyudH5Dataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = os.path.join(cfg.ROOT_DIR, opt.dataroot, opt.phase_anno)
        print('NyudH5Dataset initialize:',self.root)
        self.depth_normalize = 10.0  # the max depth is 10m

        # self.dir_anno = os.path.join(cfg.ROOT_DIR, opt.dataroot, 'annotations', opt.phase_anno + '_annotations.json')


        classes, class_to_idx = find_classes(self.root)
        imgs = make_dataset(self.root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data_size = len(self.imgs)
        self.uniform_size = (480, 640)
        self.loader=h5_loader

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth


    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data = self.online_aug(index)
        return data

    def online_aug(self, index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        # A_path = self.A_paths[anno_index]
        # B_path = self.B_paths[anno_index]

        # if self.A is None:
        #     A = cv2.imread(A_path)  # bgr, H*W*C
        #     B = cv2.imread(B_path, -1) / self.depth_normalize  # the max depth is 10m
        # else:
        #     A = self.A[anno_index]  # C*W*H
        #     B = self.B[anno_index] / self.depth_normalize # the max depth is 10m
        #     A = A.transpose((2, 1, 0))  # H * W * C
        #     B = B.transpose((1, 0))  # H * W
        #     A = A[:, :, ::-1].copy() #rgb -> bgr

        path, target = self.imgs[index]
        rgb, depth = self.loader(path)



        A = rgb[:, :, ::-1].copy() #rgb -> bgr
        B = depth / self.depth_normalize

        flip_flg, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop()

        A_resize = self.flip_pad_reshape_crop(A, flip_flg, crop_size, pad, 128)
        B_resize = self.flip_pad_reshape_crop(B, flip_flg, crop_size, pad, -1)

        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        A_resize = A_resize[::-1, :, :]

        # to torch, normalize
        A_resize = self.scale_torch(A_resize, 255.)
        B_resize = self.scale_torch(B_resize, resize_ratio)

        B_bins = self.depth_to_bins(B_resize)
        invalid_side = [int(pad[0] * resize_ratio), 0, 0, 0]

        data = {'A': A_resize, 'B': B_resize, 'A_raw': A, 'B_raw': B, 'B_bins': B_bins, 'A_paths': path,
                'B_paths': path, 'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
        return data

    def set_flip_pad_reshape_crop(self):
        """
        Set flip, padding, reshaping, and cropping factors for the image.
        :return:
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        raw_size = np.array([cfg.DATASET.CROP_SIZE[1], 416, 448, 480, 512, 544, 576, 608, 640])
        size_index = np.random.randint(0, 9) if 'train' in self.opt.phase else 8

        # pad
        pad_height = raw_size[size_index] - self.uniform_size[0] if raw_size[size_index] > self.uniform_size[0]\
                    else 0
        pad = [pad_height, 0, 0, 0]  # [up, down, left, right]

        # crop
        crop_height = raw_size[size_index]
        crop_width = raw_size[size_index]
        start_x = np.random.randint(0, int(self.uniform_size[1] - crop_width)+1)
        start_y = 0 if pad_height != 0 else np.random.randint(0,
                int(self.uniform_size[0] - crop_height) + 1)
        crop_size = [start_x, start_y, crop_height, crop_width]

        resize_ratio = float(cfg.DATASET.CROP_SIZE[1] / crop_width)

        return flip_flg, crop_size, pad, resize_ratio

    def flip_pad_reshape_crop(self, img, flip, crop_size, pad, pad_value=0):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                       constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        # Crop the resized image
        img_crop = img_pad[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # Resize the raw image
        img_resize = cv2.resize(img_crop, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        return img_resize

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img, scale):
        """
        Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W
        """
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
        else:
            img = transforms.Normalize((0,), (1,))(img)
        return img

    def name(self):
        return 'nyud_h5'


