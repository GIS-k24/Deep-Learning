from torchvision import models, transforms
import torchvision
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data = torchvision.datasets.MNIST(
    root='D:\编程\python\PyTorch\img',
    train=True,
    transform=transform,
    download=False
)

dataloader = torch.utils.data.DataLoader(
    dataset=mnist_data,
    batch_size=batch_size,
    shuffle=True,
)

image_size = 28 * 28
hidden_size = 256

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

latent_size = 64

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

D = D.to(device)
G = G.to(device)

loss_fn = nn.BCELoss()  # 交叉熵
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

total_steps = len(dataloader)
num_epochs = 30

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.shape[0]
        images = images.reshape(batch_size, image_size).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs  # 对于D来说，越大越好

        # 开始生成fake images
        z = torch.randn(batch_size, latent_size).to(device)  # latent variable
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs  # 对于D来说，越小越好

        # 开始优化 discriminator
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels)

        # 开始优化 generator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 200 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}"
                    .format(epoch,
                            num_epochs,
                            i,
                            total_steps,
                            d_loss.item(),
                            g_loss.item(),
                            real_score.mean().item(),
                            fake_score.mean().item()))

z = torch.randn(batch_size, latent_size).to(device)  # latent variable
fake_images = G(z)
fake_images = fake_images.view(batch_size, 28, 28).data.cpu().numpy()
plt.imshow(fake_images[0], cmap="gray")
