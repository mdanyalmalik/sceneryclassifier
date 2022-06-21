# Scenery Classifier

A scenery classifier made using a CNN in pytorch, with the help of the intel image dataset from kaggle.

<a href ="https://huggingface.co/spaces/danyalmalik/sceneryclassifier">VIEW DEMO</a>

## Description

Through the use of 2d convolution operations, a convolutional neural network is constructed and trained. The model weights are stored in a .pth file. This model is loaded in the gradio interface to categorise images into the 6 categories, buildings, forest, glacier, mountains, sea and street.

## Getting Started

- to view the demo, simply click the link above

### Dependencies

- torch
- torchvision
- numpy
- tqdm
- gradio

### Executing program

- After downloading the files, create a new folder named 'data' in the 'train' directory
- Next, download the 'seg_train' and 'seg_test' folders from <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">here</a> .
- Run train.py inside the 'train' directory. (You might have to make a new directory 'models' in the main directory)
- Run ui.py inside the 'interface' directory, then click the link in the console.

## Authors

Me: [@mdanyalmalik]

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
