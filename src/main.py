import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from process_data import unpickle, cifar10
import torch.optim as optim
from CNN import ConvNet
from torch.utils.data.sampler import SubsetRandomSampler


CIFAR_DIR = 'Data/'

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

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

data_1, label_1 = data_batch1[b'data'].reshape((10000, 3, 32, 32)), batch_1[b'labels']
data_2, label_2 = data_batch2[b'data'].reshape((10000, 3, 32, 32)), batch_2[b'labels']
data_3, label_3 = data_batch3[b'data'].reshape((10000, 3, 32, 32)), batch_3[b'labels']
data_4, label_4 = data_batch4[b'data'].reshape((10000, 3, 32, 32)), batch_4[b'labels']
data_5, label_5 = data_batch5[b'data'].reshape((10000, 3, 32, 32)), batch_5[b'labels']
test_data, test_labels = test_batch[b'data'].reshape((10000, 3, 32, 32)), test_batch[b'labels']

#Training the model(Hyperparameters)
num_epochs = 2
model = ConvNet().to(device)
batch_size = 64
learning_rate = 0.001
valid_size = 0.2 # percentage of training set to use as validation
valid_loss = 0.0
train_loss = 0.0
valid_loss_min = np.Inf # track change in validation loss

#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#dataset has PILImage images of range [0,1]
#We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_arr = np.concatenate((data_1, data_2, data_3, data_4, data_5), axis=0)
train_arr = train_arr.transpose(0, 2, 3, 1).astype(np.float32)
train_labels = label_1 + label_2 + label_3 + label_4 + label_5
train_labels = np.array(train_labels)
test_data = test_data.transpose(0, 2, 3, 1).astype(np.float32)
test_labels = np.array(test_labels)

train_labels = torch.LongTensor(train_labels)
train_data = cifar10(train_arr, train_labels)

#obtaining training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


#Define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


#Prepare dataloaders 
train_loader = DataLoader(dataset=train_data, batch_size = batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset = train_data, batch_size = batch_size, sampler=valid_sampler)

test_data = cifar10(test_data, test_labels)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


n_total_steps = len(train_loader)
# n_total_steps = number of batches i.e 50,000/4 where 4 is the batch size

for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        #original shape: [4, 3, 32, 32] = 4, 3, 1024
        data = data.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data.size(0)

        if (i+1) % 2000 ==0:
            print (f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    model.eval()
    for data, labels in valid_loader:
        outputs = model(data)
        # calculate the batch loss
        loss = criterion(outputs, labels)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
print('Finished Training')

#Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range (10)]
    n_class_samples = [0 for i in range(10)]
    for data, labels in test_loader:
        batch_size = data.size(0)
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        
        #max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
