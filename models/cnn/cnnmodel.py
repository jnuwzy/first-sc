
import torch
from torch import optim
import cifar
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

input_size = 3072  # 3*32*32
hidden_size1 = 500  # 第一次隐藏层个数
hidden_size2 = 200  # 第二次隐藏层个数
num_classes = 10  # 分类个数
num_epochs = 5  # 批次次数
learning_rate = 1e-3
batch_size = 1000  #批次大小

class LeNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(LeNet, self).__init__()
        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.shape)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_gpu = LeNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net_gpu.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print('current epoch + %d' % epoch)
        running_loss = 0.0
        for i, (images, labels) in enumerate(cifar.train_loader, 0):
            images = images.to(device)
            labels = labels.to(device)

            # images = images.view(images.size(0), -1)
            labels = labels.clone().detach().long()

            # 梯度清零
            optimizer.zero_grad()
            outputs = net_gpu(images)  # 将数据集传入网络做前向计算
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                      % (epoch + 1, i, running_loss))
                running_loss = 0.0
    print('Finished Training')

    # prediction
    total = 0
    correct = 0
    acc_list_test = []
    for images, labels in cifar.test_loader:
        # images = images.view(images.size(0), -1)
        images = images.to(device)
        labels = labels.to(device)
        outputs = net_gpu(images)  # 将数据集传入网络做前向计算

        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
        acc_list_test.append(100 * correct / total)

    acc_list_test_cpu = [item.cpu() for item in acc_list_test]  # 将列表中的每个张量移到 CPU 上
    acc_list_test_numpy = [item.numpy() for item in acc_list_test_cpu]  # 将列表中的每个张量转换为 NumPy 数组

    print('Accuracy = %.2f' % (100 * correct / total))
    plt.plot(acc_list_test_numpy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.savefig('Accuracy On TestSet.png')
    plt.show()