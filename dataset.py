import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')
class prepare_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform = None, need_img_path=False):

        self.data_path = data_path
        self.transform = transform
        self.cls_num = 2
        self.img_path = []
        self.labels = []
        self.need_img_path = need_img_path
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if root.split('/')[-1] == 'NORMAL':
                    self.img_path.append(os.path.join(root, file))
                    self.labels.append(0)
                # elif root.split('/')[-1] == 'ABNORMAL':
                else:
                    if data_path.split('/')[-1] == 'train':
                        continue
                    self.img_path.append(os.path.join(root, file))
                    self.labels.append(1)
        self.CLASSES = ['NORMAL', 'ABNORMAL']
        
    def get_cls_num_list(self):
        cls_num_list = []
        for tempLabel in range(self.cls_num):
            cls_num_list.append(np.sum(np.array(self.labels)==tempLabel))
        return cls_num_list
    
    def __getitem__(self, index):

        imgName = self.img_path[index]
        Img = Image.open(imgName)
        if Img.mode != 'RGB':
            Img = Img.convert('RGB')
        # Img = cv2.imread(imgName)
        # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        if self.transform is not None:
            Img = self.transform(Img)

        # Get the Labels
        label = self.labels[index]
        label_onehot = np.zeros(5)
        label_onehot[label] = 1
            
        if self.need_img_path:
            return Img, label, label_onehot, imgName
        else:
            return Img, label, label_onehot

    def __len__(self):
        return len(self.labels)
