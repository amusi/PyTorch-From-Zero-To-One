# Summary：PyTorch0.4.0，而且是在Windows上
# Author:  Amusi
# Date:    2018-03-31
# Reference: https://zhuanlan.zhihu.com/p/39667289

# 导入包
# torch: Pytorch最核心的库（可以代表PyTorch），主要用于计算 
# torchvision，其实PyTorch常用到的库，有models、datasets和transforms三个子包，分别提供了网络、预训练模型、数据集和格式变换等功能
# torchvision: https://github.com/pytorch/vision/tree/master/torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# cifar-10官方提供的数据集是用numpy array存储的
# 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据，这里我们使用CIFAR数据集，大概200MB，网速不好的同学，请耐心等待（并不是程序卡住了）
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          transform=transform)

										 
print(cifar_train)

print(cifar_test)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)

class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))	
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
		
# 如果你没有GPU，那么可以忽略device相关的代码
device = torch.device("cuda:0")
net = LeNet().to(device)

# optim中定义了各种各样的优化方法，包括SGD
import torch.optim as optim

# CrossEntropyLoss就是我们需要的损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Start Training...")
for epoch in range(30):
    # 我们用一个变量来记录每100个batch的平均loss
    loss100 = 0.0
    # 我们的dataloader派上了用场
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 注意需要复制到GPU
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss100 += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss100 / 100))
            loss100 = 0.0

print("Done Training!")

# 构造测试的dataloader
dataiter = iter(testloader)
# 预测正确的数量和总数量
correct = 0
total = 0
# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 预测
        outputs = net(images)
        # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))