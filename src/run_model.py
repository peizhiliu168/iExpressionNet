###################################################
# define functions here to train, test CNN models #
###################################################

from models import Emotion_Classifier_Conv
import torch
import torch.nn as nn
import torch.optim as optim
from ingest import ingest_fer13
from fer13 import FER2013
from torch.utils.data import Dataset, DataLoader




use_cuda = torch.cuda.isavailable()
net = Emotion_Classifier_Conv()

criterion = nn.CrossEntropyLoss()
lr = 0.01
bs = 128
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# path to data file
path = None
train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels = ingest_fer13(path)

trainset = FER2013('Training', train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)


def train(epoch):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    Train_acc = 100.*correct/total
