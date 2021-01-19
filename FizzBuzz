"""
FizzBuzz 游戏，规则如下：
从 1 开始往上数，当遇到 3 的倍数时，说 fizz，当遇到 5 的倍数时，说buzz， 当遇到 15 的倍数时，说 fizzbuzz，其他情况说正常数
------------------------------------------------
1、按常规编程的策略
"""


# 【1】
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))


"""
for i in range(1, 16):
    helper(i)
"""

import numpy as np
import torch
import torch.tensor as tensor

NUM_DIGHTS = 10


def binary_encode(i, num_dights):
    return np.array([i >> d & 1 for d in range(num_dights)][::-1])


trX = torch.Tensor([binary_encode(i, NUM_DIGHTS) for i in range(101, 2 ** NUM_DIGHTS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGHTS)])

# print(trX.shape)  # [923, 10]
# print(trY.shape)  # [923,]

NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(10, NUM_HIDDEN),  # [923, 10] * [10, 100] = [923, 100]
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4),  # [923, 100] * [100, 4] = [923, 4] | 4 -> fizz_buzz_encode 的 [0, 1, 2, 3]
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.05)

if torch.cuda.is_available():
    model = model.cuda()

BATCH_SIZE = 128

"""

for epoch in range(1000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        # forward pass
        y_pred = model(batchX)

        # loss function
        loss = loss_fn(y_pred, batchY)
        print("Epoch:", epoch, "Loss:", loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

PATH = r"D:\编程\python\PyTorch\thesis\Models" + "\Code_03_" + "model_Adam" + ".tar"

"""

# 模型保存 
torch.save(model, PATH)
"""

model = torch.load(PATH)

textX = torch.Tensor([binary_encode(i, NUM_DIGHTS) for i in range(1, 101)])

if torch.cuda.is_available():
    textX = textX.cuda()

with torch.no_grad():
    testY = model(textX)

tYL = testY.max(1)[1].data.numpy().tolist()

prediction = zip(range(1, 101), tYL)

print([fizz_buzz_decode(i, x) for i, x in prediction])
