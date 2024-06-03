import numpy as np
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
import torchvision.datasets as dsets

batch_size = 100
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
])

# Cifar110 dataset
train_dataset = dsets.CIFAR10(root='/ml/pycifar',
                              train=True,
                              download=True,
                              transform=transform
                              )
test_dataset = dsets.CIFAR10(root='/ml/pycifar',
                             train=False,
                             download=True,
                             transform=transform
                             )
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True
                                          )

# import matplotlib.pyplot as plt
# fig = plt.figure()
# classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     (_, label) = train_dataset[i]
#     plt.imshow(train_loader.dataset.data[i],cmap=plt.cm.binary)
#     plt.title("Labels: {}".format(classes[label]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()