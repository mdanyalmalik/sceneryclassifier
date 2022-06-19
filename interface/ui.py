import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import os

from net import Net

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


net = Net()
net.to(device)

model = "1655643142.677759_Acc0.59_modelweights.pth"
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
