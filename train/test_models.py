import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from test import test


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


device = torch.device("cuda:0")
net = Net()
net.to(device)

test_mode = 1  # 0 for single model, 1 for comparison

# enter model filename here (for single model mode)
model1 = "1655321975.6330233_Acc0.647_modelweights.pth"

test_runs = 10  # number of times to test and avg

if test_mode == 0:
    acc = 0
    net.load_state_dict(torch.load(os.path.join('../models/', model1)))
    for test_run in range(test_runs):
        acc += test(net, device)
    acc /= test_runs
    print('Avg acc:', acc)

elif test_mode == 1:
    acc = 0
    models = {}
    for model in os.listdir('../models/'):

        net.load_state_dict(torch.load(os.path.join('../models/', model1)))
        for test_run in range(test_runs):
            acc += test(net, device, print_acc=False)
        acc /= test_runs

        print(f"Model: {model}\nAvg acc: {acc}\n")

        models.update({model: acc})

    best_model = sorted(models, key=models.get)[-1]
    print("\nBest Model:", best_model)
