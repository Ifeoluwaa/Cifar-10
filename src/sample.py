import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from process_data import unpickle, cifar10
import torch.optim as optim
from torch.utils.data import Dataset
from CNN import ConvNet
import matplotlib.pyplot as plt


#file path
CIFAR_DIR = 'Data/'
def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

batch_1 = data_batch1
batch_2 = data_batch2
batch_3 = data_batch3
batch_4 = data_batch4
batch_5 = data_batch5
test_batch = test_batch



#print(all_data, test_batch)


#print(batch_meta)
#print(data_batch1.keys())
#print(data_batch1[b'labels'])
#print(len(data_batch1[b'data']))
print(len(data_batch1[b'data'][0]))



class cifar10(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        transform_to_tensor = transforms.ToTensor()
        return transform_to_tensor(self.data[idx]), self.labels[idx]

#print(type(all_data))
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

num_epochs = 2
#model = ConvNet().to(device)
batch_size = 64
learning_rate = 0.001

data_1, label_1 = data_batch1[b'data'].reshape((10000, 3, 32, 32)), batch_1[b'labels']
data_2, label_2 = data_batch2[b'data'].reshape((10000, 3, 32, 32)), batch_2[b'labels']
data_3, label_3 = data_batch3[b'data'].reshape((10000, 3, 32, 32)), batch_3[b'labels']
data_4, label_4 = data_batch4[b'data'].reshape((10000, 3, 32, 32)), batch_4[b'labels']
data_5, label_5 = data_batch5[b'data'].reshape((10000, 3, 32, 32)), batch_5[b'labels']
test_data, test_labels = test_batch[b'data'].reshape((10000, 3, 32, 32)), test_batch[b'labels']

train_arr = np.concatenate((data_1, data_2, data_3, data_4, data_5), axis=0)
train_arr = train_arr.transpose(0, 2, 3, 1).astype(np.float32)
train_labels = label_1 + label_2 + label_3 + label_4 + label_5
train_labels = np.array(train_labels)
test_data = test_data.transpose(0, 2, 3, 1).astype(np.float32)
test_labels = np.array(test_labels)

train_labels = torch.LongTensor(train_labels)
train_data = cifar10(train_arr, train_labels)
train_loader = DataLoader(dataset=train_data, batch_size =batch_size, shuffle=True)

test_labels = torch.LongTensor(test_labels)
test_data = cifar10(test_data, test_labels)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print(f"Sizes of train_dataset: {len(train_data)} and test_dataet: {len(test_data)}")
print(f"Sizes of train_loader: {len(train_loader)} and test_loader: {len(test_loader)}")

print(f"len(train_loader) = {len(train_loader)} & len(test_loader) = {len(test_loader)}")

 #Get some random batch of training images & labels-
data, labels = next(iter(train_loader))

# You get 64 images due to the specified batch size-
print(f"images.shape: {data.shape} & labels.shape: {labels.shape}")

# Sanity check-
print(len(train_data) / batch_size, len(test_data) / batch_size)

print(data.shape)

print (train_arr[10000].shape)
print(labels.size())

print(len(train_loader.dataset))
print(labels.shape)
