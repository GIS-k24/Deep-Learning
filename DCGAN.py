from torchvision import models, transforms
import torchvision
import torch
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 64
batch_size = 128
dataroot = r""
num_workers = 0

"""
dataset = torchvision.datasets.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ]))
"""

dataset = torchvision.datasets.CIFAR10(
    root='D:\编程\python\PyTorch\img',
    train=False,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ]))

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

"""
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis = ("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True)))
"""


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


nz = 100
ngf = 64
nc = 3  # RGB


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 【6】100 1 1
            nn.ConvTranspose2d(  # 反向操作 | 从下往上倒着看
                in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 【5】512 4 4
            nn.ConvTranspose2d(
                in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 【4】256 8 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 【3】128 16 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 【2】64 32 32
            nn.ConvTranspose2d(
                in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
            # 【1】 3 64 64
        )

    def forward(self, x):
        x = self.main(x)
        return x


G = Generator().to(device)
G.apply(weight_init)

ndf = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 【1】3 64 64
            nn.Conv2d(
                in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 【2】64 32 32
            nn.Conv2d(
                in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 【3】128 16 16
            nn.Conv2d(
                in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 【4】256 8 8
            nn.Conv2d(
                in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 【4】512 4 4
            nn.Conv2d(
                in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=2, padding=0
            ),
            # 【5】1 1 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


D = Discriminator().to(device)
D.apply(weight_init)

loss_fn = nn.BCELoss()  # 交叉熵
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=[0.5, 0.999])
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=[0.5, 0.999])

total_steps = len(dataloader)
num_epochs = 5

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        D.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label = torch.ones(b_size).to(device)
        output = D(real_images).view(-1)

        real_loss = loss_fn(output, label)
        real_loss.backward()
        D_x = output.mean().item()

        # 生成假图片
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = G(noise)  # 输出结果为 None ???
        label.fill_(0)
        output = D(fake_images.detach()).view(-1)
        fake_loss = loss_fn(output, label)
        fake_loss.backward()
        D_G_z1 = output.mean().item()
        loss_D = real_loss + fake_loss
        d_optimizer.step()

        # 训练generator
        G.zero_grad()
        label.fill_(1)
        output = D(fake_images).view(-1)
        loss_G = loss_fn(output, label)
        loss_G.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()

        if i % 5 == 0:
            print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}"
                .format(
                epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2
            ))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
with torch.no_grad():
    fake = G(fixed_noise).detach().cpu()

real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(30, 30))
plt.subplot(1, 2, 1)
plt.axis = ("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis = ("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
plt.show()
