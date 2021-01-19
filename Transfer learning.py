import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import time
import copy
import os

"""
迁移学习：
【1】fine tuning：从一个预训练模型开始，我们改变一些模型的架构，然后继续训练整个模型的参数（例如将最后的全连接层1000分类改为10分类）

【2】feature extraction：不改变训练模型的参数，只更新改变过的部分模型参数。把预训练的CNN模型当作一个特征提取模型，利用提取出来的特征来完成我们的训练任务
"""

data_dir = r"D:\编程\python\PyTorch\img\hymenoptera_data"  # 数据集位置
model_name = "resnet"
num_classes = 2  # 是二分类问题： 区分蚂蚁和蜜蜂
batch_size = 32
num_epochs = 15
feature_extract = True
input_size = 224

all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
    transforms.RandomResizedCrop(input_size),  # 从图片中截取 input_size * input_size 大小的部分
    transforms.RandomHorizontalFlip(),  # 图片旋转
    transforms.ToTensor(),
]))
loader = torch.utils.data.DataLoader(
    dataset=all_imgs,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),  # 在中间截取图片的一部分
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ["train", "val"]}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   ) for x in ["train", "val"]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
for step, (x, y) in enumerate(loader):
    print(step)
    plt.imshow(x[0][0].numpy(), cmap='gray')
    plt.show()
    print(y[0])
"""

"""
img = next(iter(loader))[0]
print(img)
"""

unloader = transforms.ToPILImage()
plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1)


# plt.figure()
# imshow(img[0], title="Image")



def set_parameter_requires_grad(model, feature_extract):
    # 如果不是 fine tuning，就可以把 grad 设为 False
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


"""
model_ft = models.resnet18()
print(model_ft.fc)
print(model_ft.fc.in_features)
"""


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)  # if true -> a trained model else -> a random model
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features  # 将原始的 1000分类 问题变为 指定数值 分类问题
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("model not implemented")
        return None, None

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# print(model_ft)

"""
# 除了最后的全连接层的 梯度 有变化，其他的都没有
print(model_ft.layer1[0].conv1.weight.requires_grad)
print(model_ft.fc.weight.requires_grad)
"""


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.0
            running_correct = 0.0
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, lables in dataloaders[phase]:
                inputs, lables = inputs.to(device), lables.to(device)

                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, lables)

                pred = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(pred.view(-1) == lables.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct / len(dataloaders[phase].dataset)

            print("Phase {}, loss:{}, acc:{}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# 使用预训练的模型
model_ft = model_ft.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
ft_model, ft_history = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)

# 未使用预训练的模型
model_scratch, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
model_scratch = model_scratch.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_scratch.parameters()), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
scratch_model, scratch_history = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)

plt.title("Validation Accuracy vs Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, 1 + num_epochs), ft_history, lable="Pretrained")
plt.plot(range(1, 1 + num_epochs), scratch_history, lable="Scratch")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.legend()
plt.show()
