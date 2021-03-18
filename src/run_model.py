###################################################
# define functions here to train, test CNN models #
###################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from .models import Emotion_Classifier_Conv
from .ingest import ingest_fer13
from .dataloaders import FER2013

# A general run model function. Can be used to both train the data as well 
# perform evaluation on the model.
def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True, device=torch.device('cpu')):

    if train_set:
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    if valid_set:
        validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    if test_set:
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    if running_mode == 'train':
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        if valid_set:
            prev_valid_loss_val = np.inf
            counter = 0
            while True:
                counter +=1 
                model, loss, acc = _train(model, trainloader, optimizer, device)
                train_loss.append(loss)
                train_acc.append(acc)

                optimizer.zero_grad()
                valid_loss_val, valid_acc_val, _ = _test(model, validloader, device)
                valid_loss.append(valid_loss_val)
                valid_acc.append(valid_acc_val)

                print("Training epoch: {}, train accuracy: {}, train loss: {}, valid accuracy: {}, valid loss: {} ".format(counter, acc, loss, valid_acc_val, valid_loss_val))
                
                if abs(prev_valid_loss_val - valid_loss_val) < stop_thr or counter >= n_epochs:
                    break
                prev_valid_loss_val = valid_loss_val
        
        else:
            for i in range(n_epochs):
                model, loss, acc = _train(model, trainloader, optimizer, device)
                train_loss.append(loss)
                train_acc.append(acc)
                print("Training epoch: {}, accuracy: {}, loss: {} ".format(i, acc, loss))

        return model, {'train':np.array(train_loss), 'valid':np.array(valid_loss)}, {'train':np.array(train_acc), 'valid':np.array(valid_acc)}
    
    elif running_mode == 'test':
        loss, accuracy, predictions = _test(model, testloader, device)
        return loss, accuracy, predictions

# private method called by run_model to train the model
def _train(model, data_loader, optimizer, device=torch.device('cpu')):

    model.train()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # zero out gradient
        optimizer.zero_grad()

        # compute gradient and step
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print("Batch {} loss: {}, accuracy: {}".format(batch_idx, loss.item(), correct/total))

    train_acc = 100. * correct / total
    return model, train_loss / len(data_loader), train_acc

# private method called by run_model to validate and test the model
def _test(model, data_loader, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 0

    predictions = torch.tensor([])
    labels = torch.tensor([])

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute gradient and step
        inputs, targets = Variable(inputs),Variable(targets)
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.long())

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        predictions = torch.cat((predictions, predicted), 0)
        labels = torch.cat((labels, targets), 0)

    train_acc = 100. * correct / total
    return train_loss / len(data_loader), train_acc, [labels.cpu().numpy(), predictions.cpu().numpy()]