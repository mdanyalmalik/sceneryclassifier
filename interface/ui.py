import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import random
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


net = models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 6)
net = net.to(device)

model = "1655988285.9725637_Acc0.88_modelweights.pth"
net.load_state_dict(torch.load(os.path.join('../models/', model)))
net.eval()

labels = ['Buildings', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Street']

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

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
        img = data_transforms(img)
        img = img.to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = net(img)
            output = F.softmax(output, dim=1)

            pred = [output[0][i].item() for i in range(len(labels))]
            print(pred)

    except Exception as e:
        pred = [0 for i in range(len(labels))]
        print(e)

    weightage = {labels[i]: pred[i] for i in range(len(labels))}
    return weightage


gr.Interface(fn=predict, inputs=gr.Image(type='pil'),
             outputs='label', title=title, examples=examples()).launch()
