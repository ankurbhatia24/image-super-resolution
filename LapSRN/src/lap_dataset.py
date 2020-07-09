"""This is just a temporary script to create the dataset only once"""
import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import random
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    scale = random.uniform(0.9, 1)
    y = y.resize((int(y.size[0]*scale), int(y.size[1]*scale)), Image.BICUBIC)
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, LR_transform=None, HR_2_transform=None, 
                                 HR_4_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.LR_transform = LR_transform
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        HR_4 = self.HR_4_transform(input)
        HR_2 = self.HR_2_transform(HR_4)
        LR = self.LR_transform(HR_4)
        to_tensor = torchvision.transforms.ToTensor()
        HR_4 = to_tensor(HR_4)
        return LR, HR_2, HR_4

    def __len__(self):
        return len(self.image_filenames)
