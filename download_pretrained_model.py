import torch
from torch.hub import load_state_dict_from_url
import os

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def download_pretrained_model():
    for model_name, model_url in model_urls.items():
        load_state_dict_from_url(model_url, model_dir='models/pytorch/imagenet/')


if __name__ == '__main__':
    download_pretrained_model()
