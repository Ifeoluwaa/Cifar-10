import torch.nn as nn
import torch.nn.functional as F
import torch

#define the CNN architecture

class ConvNet(nn.Module):
  def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
      self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
      self.pool = nn.MaxPool2d(kernel_size=2)
      self.fc1 = nn.Linear(in_features= 4 *4 *256, out_features=512)
      self.fc2 = nn.Linear(in_features=512, out_features=128)
      # Drop layer to deletes 20% of the features to help prevent overfitting
      self.drop = nn.Dropout2d(p=0.2)
      self.fc3 = nn.Linear(in_features=128, out_features=10)



  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    #print(x.shape)
     #To prevent overfitting
    #x = F.dropout(self.drop(x), training=self.training)
    #print(x.shape)
    #x = x.view(x.size(0), -1)
    #Flatten
    #x = x.view(-1,  256* 4 * 4)
    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    #return torch.log_softmax(x, dim=1)
    return x
