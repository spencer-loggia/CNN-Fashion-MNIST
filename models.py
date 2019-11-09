""" Model classes defined here! """
import math

import torch
import gc
import skimage as sk
from scipy import ndarray
from skimage import transform
from skimage import util
from skimage import filters as filt
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n1_chan, n1_kern)
        self.conv2 = nn.Conv2d(n1_chan, 10, n2_kern, stride=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(8)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 10)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self, hidden: int, channel: int, final_chan: int, compression_ratio: int, initial_kernel: int, internal_kernel: int):
        '''

        :param hidden: a list of hidden layer for fully connected section
        :param channels: a list of channel sizes (not including pooling layers!)
        :param kernels: a list of kernel sizes including pooling layers
        :param strides: a list of stride sizes including pooling layers
        '''
        super(BestNN, self).__init__()
        self.relu = nn.ReLU()
        # kernel size
        self.conv_init = nn.Sequential(nn.Conv2d(1, channel, initial_kernel, 1, 1), nn.ReLU())

        self.block1 = nn.Sequential(
            nn.Conv2d(channel, channel, internal_kernel, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(int(channel), channel, internal_kernel, 1, 1),
            nn.BatchNorm2d(channel),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(channel, int(channel*compression_ratio), internal_kernel, 1, 1),
            nn.BatchNorm2d(int(channel*compression_ratio)),
            nn.ReLU(),
            nn.Conv2d(int(channel*compression_ratio), int(channel*compression_ratio), internal_kernel, 1, 1),
            nn.BatchNorm2d(int(channel*compression_ratio)),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(int(channel*compression_ratio), int(channel*compression_ratio**2), internal_kernel, 1, 1),
            nn.BatchNorm2d(int(channel*compression_ratio**2)),
            nn.ReLU(),
            nn.Conv2d(int(channel*compression_ratio**2), int(channel*compression_ratio**2), internal_kernel, 1, 1),
            nn.BatchNorm2d(int(channel*compression_ratio**2)),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.avg_pool = nn.Sequential(
            nn.Conv2d(int(channel*compression_ratio**2), hidden, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(hidden, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )

        # self.conv1 = nn.Conv2d(1, channels[0], kernels[0], stride=strides[0], padding=paddings[0])
        # self.batch_norm1 = nn.BatchNorm2d(channels[0])
        # self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1], stride=strides[1], padding=paddings[1])
        # self.batch_norm2 = nn.BatchNorm2d(channels[1])
        # self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2], stride=strides[2], padding=paddings[2])
        # self.batch_norm3 = nn.BatchNorm2d(channels[2])
        # self.pool1 = nn.MaxPool2d(kernels[3], stride=strides[3], padding=paddings[3])
        #
        # lin1_in = self.conv_out_dim(28, kernels[:len(kernels)-2], strides[:len(strides)-2], paddings[:len(paddings)-2])
        # self.lin1 = nn.Linear((lin1_in**2)*channels[2], hidden[0])
        # self.lin2 = nn.Linear(hidden[0], hidden[1])
        # self.lin3 = nn.Linear(hidden[1], hidden[2])
        #
        # self.conv4 = nn.Conv2d(1, channels[3], kernels[4], stride=strides[4], padding=paddings[4])
        # self.batch_norm4 = nn.BatchNorm2d(channels[3])
        # self.pool2 = nn.MaxPool2d(kernels[5], stride=strides[5], padding=paddings[5])
        #
        # lin2_in = self.conv_out_dim(int(math.sqrt(hidden[2])), kernels[len(kernels) - 2:], strides[len(strides) - 2:], paddings[len(paddings) - 2:])
        # self.lin4 = nn.Linear((lin2_in**2)*channels[3], 80)
        # self.lin5 = nn.Linear(80, 30)
        # self.lin6 = nn.Linear(30, 10)

    def forward(self, x):
        ##EFF RESNET
        x = x.view(-1, 1, 28, 28)
        x = self.conv_init(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)

        x = self.final(x)

        gc.collect()
        return x

        ##BELOW GETS 90% ACC
        # x = x.view(-1, 1, 28, 28)
        #
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.batch_norm1(x)
        # x = self.conv2(x)
        # x = self.batch_norm2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.batch_norm3(x)
        # x = self.pool1(x)
        # x = self.relu(x)
        #
        # x = x.reshape(x.size(0), -1)
        #
        # x = self.lin1(x)
        # x = self.relu(x)
        # x = self.lin2(x)
        # x = self.relu(x)
        # x = self.lin3(x)
        # x = self.relu(x)
        #
        # x = x.view(-1, 1, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1))))
        #
        # x = self.conv4(x)
        # x = self.batch_norm4(x)
        # x = self.pool2(x)
        # x = self.relu(x)
        #
        #
        # x = x.reshape(x.size(0), -1)
        #
        # x = self.lin4(x)
        # x = self.relu(x)
        # x = self.lin5(x)
        # x = self.relu(x)
        # x = self.lin6(x)
        # return x

    def denoise(self, x: torch.Tensor):
        for i in range(len(x)):
            temp = x[i]
            temp = temp.reshape(temp.size(2), temp.size(1))
            temp = filt.sobel(temp)
            temp = filt.median(temp)
           # temp = filt.unsharp_mask(temp, radius=4)
            temp = torch.Tensor(temp)
            temp = temp.view(1, 1, temp.size(0), temp.size(1))
            x[i] = torch.Tensor(temp)
        return x

    def conv_out_dim(self, dim_in: int, kernels: list, strides: list, paddings: list) -> int:
        out = int((dim_in - kernels[0] + 2*(paddings[0])) / strides[0]) + 1
        if len(kernels) > 1:
            return self.conv_out_dim(out, kernels[1:len(kernels)], strides[1:len(kernels)], paddings[1:len(kernels)])
        else:
            return int(out)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)



