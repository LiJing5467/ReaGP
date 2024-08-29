import dill
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import csv
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.nn import MaxPool2d, ReLU, Linear, Flatten, Sequential, Conv2d, Dropout, Softmax, Sigmoid, BatchNorm2d, \
    AvgPool2d, AdaptiveAvgPool2d
import torch.nn.functional as F
# import torchvision
# from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from scipy.stats import pearsonr
import random
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
import statistics
import math



def deepgs(hyperparams,data_all):
    start_time = time.time()

    ####早期停止
    class EarlyStopping:
        def __init__(self, patience):
            self.patience = patience
            self.counter = 0
            self.best_loss = float('inf')
            self.early_stop = False



        def __call__(self, val_loss):
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # early_stopping = EarlyStopping(patience=150)


    class resblk(nn.Module):
        def __init__(self, ch_in, chi_out, stride):
            super(resblk, self).__init__()

            self.conv1 = Conv2d(ch_in, chi_out, 3, stride=stride, padding=1, bias=False)
            self.bn1 = BatchNorm2d(chi_out)
            self.dr = Dropout(0)
            self.relu = ReLU()

            self.conv2 = Conv2d(chi_out, chi_out, 3, stride=1, padding=1, bias=False)
            self.bn2 = BatchNorm2d(chi_out)
            self.dr = Dropout(0)
            self.extra = Sequential()
            if stride != 1 or chi_out != ch_in:
                self.extra = Sequential(
                    Conv2d(ch_in, chi_out, 1, stride=stride, bias=False),
                    BatchNorm2d(chi_out)
                    )

        def forward(self, x):
            identity = x
            out = F.relu(self.dr((self.bn1(self.conv1(x)))))
            out = self.dr(self.bn2(self.conv2(out)))
            out = out + self.extra(identity)
            out = F.relu(out)

            return out

    class se_block(nn.Module):
        def __init__(self, chi_in, ratio=16):
            super(se_block, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(chi_in, chi_in // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(chi_in // ratio, chi_in, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c,h, w, = x.size()
            y = self.avg_pool(x).view([b, c])
            y = self.fc(y).view([b, c,1, 1])
            return x * y

    class tudu(nn.Module):
        def __init__(self):
            super(tudu, self).__init__()
            self.model1 = Sequential(
                Conv2d(3, 64, 7, padding=3, stride=2, bias=True),
                BatchNorm2d(64),
                Dropout(0),
                ReLU(inplace=True),

                se_block(64),

                Conv2d(64, 64, 5, padding=3, stride=2, bias=True),
                BatchNorm2d(64),
                Dropout(0.3),
                ReLU(inplace=True),

                resblk(64, 64, 1),
                se_block(64),

                ##展平+计算
                Flatten(),
                Linear(2334784, 64),
                Dropout(0.3),
                Linear(64, 1),

                )

        def forward(self, x):
            x = self.model1(x)
            return x

    tudu = tudu().cuda()

