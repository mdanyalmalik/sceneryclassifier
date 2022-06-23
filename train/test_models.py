import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

from test import test


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


net = models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 6)
net = net.to(device)

test_mode = 1  # 0 for single model, 1 for comparison

# enter model filename here (for single model mode)
model1 = "1655321975.6330233_Acc0.647_modelweights.pth"

test_runs = 10  # number of times to test and avg

if test_mode == 0:
    acc = 0
    net.load_state_dict(torch.load(os.path.join('../models/', model1)))
    net.eval()
    for test_run in range(test_runs):
        acc += test(net, device)
    acc /= test_runs
    print('Avg acc:', acc)

elif test_mode == 1:
    models = {}
    for model in os.listdir('../models/'):
        acc = 0

        net.load_state_dict(torch.load(os.path.join('../models/', model)))
        net.eval()
        for test_run in range(test_runs):
            acc += test(net, device, print_acc=False)
        acc /= test_runs

        print(f"Model: {model}\nAvg acc: {acc}\n")

        models.update({model: acc})

    best_model = sorted(models, key=models.get)[-1]
    print("\nBest Model:", best_model)
