import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


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


net = Net()
net.to(device)

model = "1655322438.4798265_Acc0.693_modelweights.pth"
net.load_state_dict(torch.load(os.path.join('../models/', model)))

labels = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

title = "Scenery Classifier"


def examples():
    number = 8
    egs = []
    for i in range(number):
        imgs = os.listdir('examples')
        eg = imgs[random.randrange(0, len(imgs))]
        egs.append(os.path.join('examples/', eg))

    return egs


def predict(img):
    try:
        img = torch.tensor(img).view(-1, 3, 150, 150)
        img = img.to(device)
        img = img.float()
        img = img / 255.0

        with torch.no_grad():
            output = net(img)

            pred = [output[0][i].item() for i in range(len(labels))]

    except:
        pred = [0 for i in range(len(labels))]

    weightage = {labels[i]: pred[i] for i in range(len(labels))}
    return weightage


gr.Interface(fn=predict, inputs=gr.Image(shape=(150, 150)),
             outputs='label', title=title, examples=examples()).launch()
