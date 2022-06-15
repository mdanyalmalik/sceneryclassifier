import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import numpy as np
import time
import os

from test import test

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on gpu")
else:
    device = torch.device("cpu")
    print("Running on CPU")


training_data = np.load("data/training_data.npy", allow_pickle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(3, 150, 150).view(-1, 3, 150, 150)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 6)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)


net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0005)
loss_function = nn.MSELoss()


def train(net, device):
    EPOCHS = 7
    BATCH_SIZE = 100

    X = torch.tensor(np.array([i[0]
                     for i in training_data])).view(-1, 3, 150, 150)
    X = X / 255.0
    y = torch.tensor(np.array([i[1] for i in training_data]))

    X = X.to(device)
    y = y.to(device)

    for epoch in range(EPOCHS):
        for i in trange(0, len(X), BATCH_SIZE):
            batch_X = X[i:i+BATCH_SIZE].view(-1, 3, 150, 150)
            batch_y = y[i:i+BATCH_SIZE].float()

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}. Loss: {loss}")
        test(net, device)


train(net, device)
accuracy = test(net, device)

models_path = '../models/'
model_name = f"{time.time()}_Acc{accuracy}_modelweights.pth"

torch.save(net.state_dict(), os.path.join(models_path, model_name))
