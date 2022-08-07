from random import shuffle
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
import numpy as np
import time
import os

from test import test

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


data_dir = 'data/seg_train'

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# defining transformations for trainset
data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# loading trainset
image_dataset = datasets.ImageFolder(data_dir, data_transforms)
data_loader = torch.utils.data.DataLoader(
    image_dataset, batch_size=32, shuffle=True)

# altering resnet 18 model to fit our outputs
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 6)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()


def train(net, device):  # trains the model using loaded trainset
    EPOCHS = 10

    for epoch in range(EPOCHS):
        net.train()
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            net.zero_grad()

            outputs = net(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}. Loss: {loss}")
        net.eval()
        test(net, device)


train(net, device)
accuracy = test(net, device)

# saving model after training
models_path = '../models/'
model_name = f"{time.time()}_Acc{accuracy}_modelweights.pth"

torch.save(net.state_dict(), os.path.join(models_path, model_name))
