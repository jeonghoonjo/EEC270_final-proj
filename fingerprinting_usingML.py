# MLP Reference : https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
 

class LoadDataset(Dataset):
    def __init__(self, path):
        df = read_csv(path, header='infer')
        self.X = df.values[:,:-1] #read all data besides the last column(label)
        self.y = df.values[:,-1]  #read only the last column
        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.33):
        test_size = round(n_test*len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

#CNN
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 30)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
    
        return x

#MLP                
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 16)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(16, 64)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(64, 128)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()        
        self.hidden4 = Linear(128, 30)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)        
        return X


def prepare_data(path):
    dataset = LoadDataset(path)
#    test_dataset = LoadDataset(test_path)

    train, test = dataset.get_splits()
    train_loader = torch.utils.data.DataLoader(train, batch_size=75, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=30, shuffle=False)

    return train_loader, test_loader


def train_model(train_dl, model):
    training_loss = []
    epoch_loss = []
    num_epoch = 500

    criterion = CrossEntropyLoss() #change?
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epoch): #500 is enough?
        if epoch %10 == 0 : 
            print('epoch: ', epoch)
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            training_loss.append(running_loss)

        epoch_loss.append(np.mean(training_loss))
        training_loss = []
    
    return epoch_loss

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        output = model(inputs)
        output = output.detach().numpy()
        actual = targets.numpy()
        output = np.argmax(output, axis=1)
        actual = actual.reshape((len(actual), 1))
        output = output.reshape((len(output), 1))
#        print('output: ', output)
#        print('actual: ', actual)
        predictions.append(output)
        actuals.append(actual)

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc

#main

path = 'training.csv'
#test_path = 'testing.csv'

train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

print('Before Normalization')
for i in range (3):
    print(list(train_dl.dataset.__getitem__(i)))


#Normalization
#1. training data
max_train = np.amax(train_dl.dataset.__getitem__(0)[0])
min_train = np.amin(train_dl.dataset.__getitem__(0)[0])

for i in range (len(train_dl.dataset)):
    max_ = np.amax(train_dl.dataset.__getitem__(i)[0])
    min_ = np.amin(train_dl.dataset.__getitem__(i)[0])

    if max_ > max_train:
        max_train = max_
        
    if min_ < min_train:
        min_train = min_

#2. testing data
max_test = np.amax(test_dl.dataset.__getitem__(0)[0])
min_test = np.amin(test_dl.dataset.__getitem__(0)[0])

for i in range (len(test_dl.dataset)):
    max_ = np.amax(test_dl.dataset.__getitem__(i)[0])
    min_ = np.amin(test_dl.dataset.__getitem__(i)[0])

    if max_ > max_test:
        max_test = max_

    if min_ < min_test:
        min_test = min_

print('max_train: ', max_train, 'min_train: ', min_train)
print('max_test: ', max_test, 'min_test: ', min_test)

for i in range (len(train_dl.dataset)):
    for j in range(4):
        train_dl.dataset.__getitem__(i)[0][j] -= min_train
        train_dl.dataset.__getitem__(i)[0][j] /= (max_train-min_train)

for i in range (len(test_dl.dataset)):
    for j in range(4):
        test_dl.dataset.__getitem__(i)[0][j] -= min_test
        test_dl.dataset.__getitem__(i)[0][j] /= (max_test-min_test)

print('After Normalization')
for i in range (3):
    print(list(train_dl.dataset.__getitem__(i)))

model = MLP(4)
epoch_loss_ = train_model(train_dl, model)
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

plt.figure(1)
step = np.arange(len(epoch_loss_))
plt.plot(step, epoch_loss_)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

