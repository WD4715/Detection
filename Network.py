import torch
from torchvision.models import resnet50
from utils import roi_pooling, making_rois
import torch.nn as nn
import numpy as np

class ResDetection(nn.Module):
    def __init__(self, n_class):
        super(ResDetection, self).__init__()
        self.model = resnet50(pretrained = True)

        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.n_class = n_class
    def forward(self, x):
        B, C, H, W = x.shape

        resnet_output = self.model(x)
        featuremap_output = resnet_output.reshape(B, -1, H//2, W//2)

        ###
        conv1 = nn.Conv2d(featuremap_output.shape[1], 512, kernel_size = (3, 3), padding = 1).to('cuda')
        ###
        output = conv1(featuremap_output)

        ###
        conv2 = nn.Conv2d(output.shape[1], 18, kernel_size=(1, 1)).to('cuda')
        conv2_ = nn.Conv2d(output.shape[1], 36, kernel_size=(1, 1)).to('cuda')
        ###

        output1 = conv2(output)
        output2 = conv2_(output)
        output = torch.cat([featuremap_output, output1, output2], dim =1)
        rois = making_rois(output)
        output = roi_pooling(output, rois)

        B, C, H, W = output.shape

        output = output.reshape(B, -1)
        B, C = output.shape
        ###
        linear1 = nn.Linear(C, 4096)
        linear2 = nn.Linear(4096, self.n_class)
        linear3 = nn.Linear(4096, self.n_class * 4)
        ###
        output = linear1(output)
        classification = linear2(output)
        bbox = linear3(output)
        return classification, bbox



if __name__ == '__main__':
    image = torch.rand(8, 3, 128, 128).to('cuda')

    model = ResDetection(1).to('cuda')
    classification, bbox = model(image)
    print(classification.shape)
    print(bbox.shape)