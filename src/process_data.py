import pickle
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader


#file path
CIFAR_DIR = 'Data/'
def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


class cifar10(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
       
    def __getitem__(self, idx):
        transform_to_tensor = transforms.ToTensor()
        return transform_to_tensor(self.data[idx]), self.labels[idx]
