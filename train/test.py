import torch
from torchvision import transforms, datasets
import numpy as np


def test(net, device, print_acc=True):
    SIZE = 100

    data_dir = 'data/seg_test'

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    data_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    data_loader = torch.utils.data.DataLoader(
        image_dataset, batch_size=SIZE, shuffle=True)

    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        output = net(inputs)
        output = output.to(torch.device("cpu"))
        total += len(output)

        for i, label in enumerate(labels):
            if (torch.argmax(output[i]) == label):
                correct += 1

    if print_acc:
        print("Acc:", round(correct/total, 3))
    return round(correct/total, 3)
