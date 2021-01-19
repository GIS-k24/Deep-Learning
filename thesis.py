from __future__ import division
from torchvision import models, transforms
import torchvision
import torch
from PIL import Image
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_size = 400


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),

])

content = load_image(r"D:\编程\python\PyTorch\img\T_content.jpg", transform, max_size=max_size)
style = load_image(r"D:\编程\python\PyTorch\img\T_style.jpg", transform, shape=[content.size(2), content.size(3)])

"""
print(content.shape)
print(style.shape)
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


plt.figure()


# imshow(style[0], title="Image")


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ["0", "5", "10", "19", "28"]
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        # _modules 可以把每层的信息取出
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


# vgg = models.vgg19(pretrained=True)

"""
# 已训练好的 VGG 不需要分类器，从特征层中取出某些需要的部分:
第 0 层 | 第 5 层 | 第 10 层 | 第 19 层 | 第 28 层
上述层可以提取风格
"""

vgg = VGGNet().to(device).eval()  # 在 test 时使用 eval() | 此处的 VGG 是预训练后的模型 | eval()后就不再优化

features = vgg(content)

"""
for feat in features:
    print(feat.shape)
"""

target = content.clone().requires_grad_(True)  # 从原始的 content 上进行梯度变换修改内容
optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])  # 优化 target 图片

num_steps = 100

for step in range(num_steps):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)

    # content loss
    # style  loss

    content_loss = style_loss = 0.
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss += torch.mean((f1 - f2) ** 2)

        _, c, h, w = f1.size()
        f1 = f1.view(c, h * w)  # [c, h * w]
        f3 = f3.view(c, h * w)

        f1 = torch.mm(f1, f1.t())  # [c, c]
        f3 = torch.mm(f3, f3.t())

        style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)

    loss = content_loss + style_loss * 100.

    # 更新 target 的 optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        print("Step [{}/{}], Content Loss:{:.4f}, Style Loss:{:.4f}".format(step, num_steps, content_loss.item(),
                                                                            style_loss.item()))

# 逆标准化
denorm = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
img = target.clone().squeeze(0)
img = denorm(img).clamp_(0, 1)
# imshow(img, title="Target Image")

img = img.detach().numpy() * 255
R = Image.fromarray(img[0].astype(np.uint8))
G = Image.fromarray(img[1].astype(np.uint8))
B = Image.fromarray(img[2].astype(np.uint8))
img = Image.merge('RGB', (R, G, B))
img.save(r"D:\编程\python\PyTorch\img\Target.jpg")
