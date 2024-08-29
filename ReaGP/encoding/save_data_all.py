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
from sklearn.model_selection import KFold
# 定义样本
# sample1 = [[0,1,1,2],[0,1,1,2],[0,1,1,2]]
# ###第三通道的数据
# target = [[20, 10, 10, 20],[20, 10, 10, 20],[20, 10, 10, 20]]
import dill
phe=[]


# 读取 总phe 文件
with open('/media/user/sda14/home/jli/DeepGS_new/cow/FDG/FDG_phe.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        phe.append(row)

# 将数据转换为二维列表
phe = np.array(phe).astype(float)
# print(bbb)


sample1=[]
target=[]
with open('/media/user/sda14/home/jli/DeepGS_new/cow/FDG/FDG_geno_snp.csv', 'r')  as file:
    reader = csv.reader(file)
    for row in reader:
        sample1.append([int(value) for value in row])

with open('/media/user/sda14/home/jli/DeepGS_new/cow/FDG/FDG_snp_all_modified_SNP_freq.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        target.append([float(value) for value in row])


# 转换函数 将 0，1，2转为one-hot
def convert_sample(sample):
    converted_sample = []
    for digit in sample:
        if digit == 0:
            converted_sample.append([0, 0])
        elif digit == 1:
            converted_sample.append([1, 0])
        elif digit == 2:
            converted_sample.append([1, 1])
    return converted_sample

def transform_array(arr):
    result = [[arr[i][j] for i in range(len(arr))] for j in range(len(arr[0]))]
    return result

####压缩（1，3，根号snp，根号snp，）
def reshap_w(tensor):

    # 确定根号长度
    sqrt_length = int(tensor.size(1) ** 0.5)
    # 转换形状
    # reshaped_tensor = tensor.resize_(1, 3, 2, 2)
    if sqrt_length*sqrt_length !=tensor.size(1):
        new_w=sqrt_length+1
        padding = (new_w * new_w - tensor.size(1)) // 2
        if (new_w*new_w-tensor.size(1))%2==0:
            reshaped_tensor=F.pad(tensor,[padding,padding,0,0])
        else:
            reshaped_tensor = F.pad(tensor, [padding, padding+1, 0, 0])
        reshaped_tensor=reshaped_tensor.reshape(1,3,new_w,new_w)
    else:
        reshaped_tensor = reshaped_tensor.reshape(1, 3, sqrt_length, sqrt_length)
    return reshaped_tensor

def san(sample1,target):


    # 将样本转换为所需格式
    converted_sample1 = convert_sample(sample1)
    arr = converted_sample1
    result = transform_array(arr)
    ###前两个通道已经变成了tensor
    tensor = torch.tensor(result)
    # print(tensor)
    # print(tensor.shape)
    ####将前两个通道和第三个通道合并，成为三通道tensor
    target = torch.tensor([target])
    merged_tensor = torch.cat((tensor, target), dim=0)
    # print(merged_tensor)
    # print(merged_tensor.shape)
    merged_tensor=reshap_w(merged_tensor)
    ####压缩成固定的形状
    compressed_tensor = merged_tensor
    return compressed_tensor


torch_snp=[]
merged_tensor = torch.empty(0)
for s, t in zip(sample1, target):
    # print(san(s,t))
    merged_tensor=torch.cat((merged_tensor,san(s,t)),dim=0)

# print(merged_tensor)

device=torch.device("cuda")

#
#
####设置dataset格式
class myDataset(Dataset):
    def __init__(self):
        # 创建snp的数据集
        self.data =merged_tensor
        # 创建snp的标签（表型）
        self.label = torch.tensor(phe)

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)



data_all_train = myDataset()


with open('/media/user/sda14/home/jli/DeepGS_new/cow/FDG/FDG_dataset_save_all.pkl','wb') as f:
    dill.dump(data_all_train, f)
    print("训练集的dataset_数据集保存完成！")
