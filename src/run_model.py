###################################################
# define functions here to train, test CNN models #
###################################################

from models import Emotion_Classifier_Conv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ingest import ingest_fer13
from fer13 import FER2013
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable



use_cuda = False#torch.cuda.is_available()
net = Emotion_Classifier_Conv()

criterion = nn.CrossEntropyLoss()
lr = 0.01
bs = 128
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# path to data file
path = "icml_face_data.csv"
train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels = ingest_fer13(path)

# load training set
trainset = FER2013('Training', train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)
# load public test set
publicTestset = FER2013('PublicTest', train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels)
publicTestloader = torch.utils.data.DataLoader(publicTestset, batch_size=bs, shuffle=False, num_workers=1)
# load private test set
privateTestset = FER2013('PrivateTest', train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels)
privateTestloader = torch.utils.data.DataLoader(privateTestset, batch_size=bs, shuffle=False, num_workers=1)


def train(epoch):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs),Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    train_acc = 100. * correct / total
    print("Training accuracy for epoch " + str(epoch) + " is " + str(train_acc))

def publicTest(epoch):

    net.eval()
    publicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(publicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        publicTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    publicTest_acc = 100. * correct / total
    print("Public Test accuracy for epoch " + str(epoch) + " is " + str(publicTest_acc))

def privateTest(epoch):

    net.eval()
    privateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(privateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        privateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    privateTest_acc = 100. * correct / total
    print("Private Test accuracy for epoch " + str(epoch) + " is " + str(privateTest_acc))


if __name__ == "__main__":
    for epoch in range(50):
        train(epoch)
        publicTest(epoch)
        privateTest(epoch)