import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def test(net, device):
    SIZE = 150

    testing_data = np.load("data/testing_data.npy", allow_pickle=True)
    np.random.shuffle(testing_data)
    testing_data = testing_data[:SIZE]

    X = torch.tensor(np.array([i[0]
                     for i in testing_data])).view(-1, 3, 150, 150)
    X = X / 255.0
    y = np.array([i[1] for i in testing_data])

    X = X.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        output = net(X.view(-1, 3, 150, 150))
        output = output.to(torch.device("cpu"))
        total += len(output)

        for i in range(len(y)):
            if (np.eye(6)[torch.argmax(output[i])] == y[i]).all():
                correct += 1

    print("Acc:", round(correct/total, 3))
    return round(correct/total, 3)
