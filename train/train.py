import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import numpy as np
import time
import os

from test import test
from net import Net

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on gpu")
else:
    device = torch.device("cpu")
    print("Running on CPU")


training_data = np.load("data/training_data.npy", allow_pickle=True)

net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0005)
loss_function = nn.MSELoss()


def train(net, device):
    EPOCHS = 10
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
