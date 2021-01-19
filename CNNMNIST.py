import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层 -- 等于一个收集器，按收集器大小为单位进行信息收集
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=5,
                stride=1,
            ),  # [28 * 28] --> (28 - 5 + 2 * 0) / 1 + 1 = 24 --> [24 * 24]
            nn.MaxPool2d(
                kernel_size=2,
            ),  # [24 * 24] --> [12 * 12]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=5,
                stride=1,
            ),  # [12 * 12] --> (12 - 5 + 2 * 0) / 1 + 1 = 10 --> [8 * 8]
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),  # [8 * 8] --> [4 * 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 4 * 4 * 50)  # 等同于 reshape
        output = self.fc(x)
        return output


train_data = datasets.MNIST(
    root='D:\编程\python\PyTorch\img',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=False
)

"""
plt.imshow(mnist_data.train_data[10].numpy(), cmap='gray')
plt.show()
"""

"""
data = [d[0].data.cpu().numpy() for d in train_data]
mean = np.mean(data)
std = np.std(data)
"""

batch_size = 50

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

"""
for step, (x, y) in enumerate(train_dataloader):
    print(step)
"""

test_data = datasets.MNIST(
    root='D:\编程\python\PyTorch\img',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=False
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

"""
for step, (x, y) in enumerate(test_dataloader):
    print(step)
"""


def train(model, device, train_dataloader, loss_func, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_func(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration:{}, Loss:{}".format(
                epoch, idx, loss.item()
            ))

    return model


"""
def test(model, device, test_dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_dataloader.dataset)
    acc = correct / len(test_dataloader.dataset) * 100.
    print("Test loss:{}, Accuracy:{}".format(total_loss, acc))
"""

LR = 0.01
Momentum = 0.5
cnn = Net().to(device)

# 优化神经网络
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=Momentum)
loss_func = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0.0

Epochs = 2
for epoch in range(Epochs):
    model = train(cnn, device, train_dataloader, loss_func, optimizer, epoch)
    # test(model, device, test_dataloader)

    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_dataloader.dataset)
    acc = correct / len(test_dataloader.dataset) * 100.
    print("Test loss:{}, Accuracy:{}".format(total_loss, acc))

"""
PATH = r"D:\编程\python\PyTorch\thesis\Models" + "\Code_04_" + "MNIST" + ".tar"

# 模型保存
torch.save(model, PATH)
"""
