###################################################
# this is where we define our CNN model.          #
###################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# using VGG 16 architecture
class Emotion_Classifier_Conv(nn.Module):
    def __init__(self):
        super(Emotion_Classifier_Conv, self).__init__()

        # block 1 (1x48x48) -> (64x24x24)
        self.conv1_block1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2_block1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool_block1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_block1 = nn.BatchNorm2d(64)

        # block 2 (64x24x24) -> (128x12x12)
        self.conv1_block2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_block2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool_block2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_block2 = nn.BatchNorm2d(128)

        # block 3 (128x12x12) -> (256x6x6)
        self.conv1_block3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2_block3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_block3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool_block3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_block3 = nn.BatchNorm2d(256)

        # block 4 (256x6x6) -> (512x3x3)
        self.conv1_block4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv2_block4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv3_block4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool_block4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_block45 = nn.BatchNorm2d(512)

        # block 5 (512x3x3) -> (512x1x1)
        self.conv1_block5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv2_block5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv3_block5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool_block5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers 
        self.fc1 = nn.Linear(512, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 7)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.bn_block1(self.conv1_block1(x)))
        x = self.pool_block1(F.relu(self.bn_block1(self.conv2_block1(x))))

        x = F.relu(self.bn_block2(self.conv1_block2(x)))
        x = self.pool_block2(F.relu(self.bn_block2(self.conv2_block2(x))))

        x = F.relu(self.bn_block3(self.conv1_block3(x)))
        x = F.relu(self.bn_block3(self.conv2_block3(x)))
        x = self.pool_block3(F.relu(self.bn_block3(self.conv3_block3(x))))

        x = F.relu(self.bn_block45(self.conv1_block4(x)))
        x = F.relu(self.bn_block45(self.conv2_block4(x)))
        x = self.pool_block4(F.relu(self.bn_block45(self.conv3_block4(x))))

        x = F.relu(self.bn_block45(self.conv1_block5(x)))
        x = F.relu(self.bn_block45(self.conv2_block5(x)))
        x = self.pool_block5(F.relu(self.bn_block45(self.conv3_block5(x))))

        x = x.view(-1, 512)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        output = self.fc3(x) # no need to apply softmax since we're using cross-entropy loss

        return output





