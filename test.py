import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from pdb import set_trace as st


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
    
def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])


################################################################################################
################################################################################################


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        
        print("img_name_list_path: {}".format(img_name_list_path))
        print(type(self.img_name_list))
        print("img_name_list: {}".format(self.img_name_list))
        
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        
        print(f"name: {name}")
        print(f"name_str: {name_str}")

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        
        print(f"img_path: {get_img_path(name_str, self.voc12_root)}")
        print(imageio.imread(get_img_path(name_str, self.voc12_root)).shape)

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])
            
            print(img.shape)

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)
            
            print(img.shape)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)
            
            print(img.shape)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
            
            print(img.shape)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
            
            print(img.shape)

        return {'name': name_str, 'img': img}

    
class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        
        st()

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out




if __name__ == '__main__':
    
    train_list = "voc12/train_aug.txt"
    voc12_root = "/home/lishixuan001/ICSI/datasets/PASCAL/VOCdevkit/VOC2012"
    
    dataset = VOC12ClassificationDataset(train_list, voc12_root=voc12_root,
                                         resize_long=(320, 640), hor_flip=True,
                                         crop_size=512, crop_method="random")
    out = dataset[5]
    
    print("============== MAIN 1 ================")
    img = out["img"]
    label = out["label"]
    print(type(img))
    print(img.shape)
    print(type(label))
    print(label)

#     loader = DataLoader(dataset, batch_size=16,
#                         shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
#     loader = iter(loader)
    
#     pack = loader.next()
    
#     print("============== MAIN 2 ================")
#     img = pack["img"]
#     label = pack["label"]
#     print(type(img))
#     print(img.shape)
#     print(type(label))
#     print(label)
    
    
    
    
    