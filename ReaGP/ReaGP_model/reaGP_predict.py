from torch.utils.data import Subset

import torch
import torch
import csv
import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from torch.nn import MaxPool1d, ReLU, Linear, Flatten, Sequential, Conv1d, Dropout, Softmax, Sigmoid, BatchNorm1d, \
    AdaptiveAvgPool2d, AvgPool1d
import torch.nn.functional as F
import math
####使用GPU
device=torch.device("cuda")

####设置随机种子
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子为固定值，例如42
set_seed(42)
#



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
from torch.nn import MaxPool2d, ReLU, Linear, Flatten, Sequential, Conv2d, Dropout, Softmax, Sigmoid, BatchNorm2d, AvgPool2d
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
from sklearn.model_selection import KFold
# 定义样本
# sample1 = [[0,1,1,2],[0,1,1,2],[0,1,1,2]]
# ###第三通道的数据
# target = [[20, 10, 10, 20],[20, 10, 10, 20],[20, 10, 10, 20]]
import dill


with open('FDG_predict_data.pkl','rb')  as f:
    test_data = dill.load(f)

###人为划分数据集(1-1226)
# train_data = Subset(dataset_save, list(range(1170)))

# ####（1226-1362）
# test_data = Subset(data_all, list(range(1170, 1299)))


data_all=test_data
#
# test_dataloader = DataLoader(data_all, batch_size=1, shuffle=False)

###定义超参数空间
hyperparams={
    # 'lr':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
    # 'dropout1':[0.1,0.2,0.3,0.4,0.5],
    # 'dropout2':[0.1,0.2,0.3,0.4,0.5],
    # 'dropout3':np.linspace(0.01,0.05,5),
    'lr': 0.01,
    'dropout1': 0.3,
    'dropout2': 0.3,
    'dropout3': 0.3,
    # 'dropout3': 0.01,
    'bs':32
}


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
        b, c, h, w, = x.size()
        y = self.avg_pool(x).view([b, c])
        y = self.fc(y).view([b, c, 1, 1])
        return x * y
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class resblk(nn.Module):
    def __init__(self, ch_in, chi_out, stride):
        super(resblk, self).__init__()

        self.conv1 = Conv2d(ch_in, chi_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(chi_out)
        self.dr = Dropout(0)
        self.relu = ReLU()

        self.conv2 =Conv2d(chi_out, chi_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(chi_out)
        self.dr = Dropout(0.3)
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
            Dropout(0.3),
            ReLU(inplace=True),

            se_block(64),

            Conv2d(64, 64, 5, padding=3, stride=2, bias=True),
            BatchNorm2d(64),
            Dropout(0.3),
            ReLU(inplace=True),

            resblk(64, 64, 1),
            se_block(64),

            # AdaptiveAvgPool2d((1, 1)),

            ##展平+计算
            Flatten(),
            Linear(2334784, 64),
            Dropout(0.3),
            Linear(64, 1),

            )

    def forward(self, x):
        x = self.model1(x)
        return x

# tudu = tudu().cuda()
tudu = tudu()

aaa=[]
pheno_preidct=[]

for i in range(10):

    state_dict = torch.load("the_reaGP_hyperparams{}.pth".format(i+1),map_location=torch.device('cuda'))


    # for k,v in state_dict.items():
    #     print("k is",k)
    #     print("v is",v)

    tudu.load_state_dict(state_dict)
    with torch.no_grad():
        tudu.eval()


        # print("网络的结构如下",tudu)


        ###设置损失函数

        lossmse = nn.L1Loss()
        # lossmse.cuda()







        ###测试集数据
        test_dataloader = DataLoader(data_all, batch_size=32, shuffle=False)

        ###合并数据集

        merged_tensor_output_test = torch.empty(0)
        merged_tensor_btach1_test = torch.empty(0)



        for batch in test_dataloader:
            imgs,targets=batch
            # batch[0]=batch[0].cuda()
            # batch[1]=batch[1].cuda()

            output = tudu(batch[0].float())


            # print("原始标签：",batch[1])
            # print("预测标签：",output)

            ##合并张量

            # merged_tensor_output_test = merged_tensor_output_test.cuda()
            # merged_tensor_btach1_test = merged_tensor_btach1_test.cuda()

            merged_tensor_output_test = torch.cat((merged_tensor_output_test, output), dim=0)
            merged_tensor_btach1_test = torch.cat((merged_tensor_btach1_test, batch[1]), dim=0)

            # print("out put is ",merged_tensor_output_test,"reall is",merged_tensor_btach1_test)

        test_loss = lossmse(merged_tensor_output_test, merged_tensor_btach1_test.float())
        # print("测试集的总预测值 is ",merged_tensor_output_test.detach().flatten(), "测试集的总实际值 is", merged_tensor_btach1_test.float().flatten())

        merged_tensor_output_test = merged_tensor_output_test.detach().cpu().numpy()
        merged_tensor_btach1_test = merged_tensor_btach1_test.float().cpu().numpy()

        test_pearson = pearsonr(merged_tensor_output_test.flatten(), merged_tensor_btach1_test.flatten())[0]





        print("预测数据集的 pearson相关性为：",test_pearson,"预测数据集的 loss为：",test_loss.item())
        aaa.append(test_pearson)
        pheno_preidct.append(merged_tensor_output_test.flatten())

        print("预测的表型",merged_tensor_output_test.flatten())


sum_array = np.zeros_like(pheno_preidct[0])

# 循环相加
for array in pheno_preidct:
    sum_array += array

# 求平均值
mean_array = sum_array / len(pheno_preidct)

print("phe is the",mean_array)
print("ave is the ", np.mean(aaa))


# # 测试网络
# input = torch.ones((64, 1, 1270))  # 创建一个大小为(64, 3, 32, 32)的输入张量，表示64个3通道的32x32图像
# print(input)
#
# output=aa(input)
# print("预测的结果为：",output)
# print(output.shape)  # 打印输出张量的形状


# print(torch.tensor(aaa))
