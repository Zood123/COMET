import torch
import torch.nn as nn
from models.resnet224 import resnet50_feature
#import torchvision.models.segmentation as models

#import pytorch_lightning as pl

from torchvision import models

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        # Load a pre-trained ResNet-50 model as the backbone

        #self.backbone = models.resnet50()
        #self.backbone = models.segmentation.fcn_resnet50(pretrained=False)
        self.backbone = models.segmentation.fcn_resnet50(pretrained=False)

        self.backbone.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        # Replace the fully connected layer with a 1x1 convolution for semantic segmentation
        #self.backbone.fc = nn.Conv2d(2048, num_classes, kernel_size=1)
        
        # Create the upsampling layers
        #self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        #self.upsample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)
        
    def forward(self, x):
        #x = self.backbone(x)
        # torch.Size([16, 2048])
        #x = x.view(x.size(0), ,1,1)
        #x = self.upsample2x(x)
        #x = self.upsample8x(x)
        x = self.backbone(x)['out']
        # torch.Size([16, 1, 224, 224])
        return x



class Deeplabv3Resnet50ExplainerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(Deeplabv3Resnet50ExplainerModel, self).__init__()
        #self.explainer = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
        self.explainer = models.segmentation.lraspp_mobilenet_v3_large(num_classes=num_classes) #deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=num_classes)
        #self.explainer = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        # weights='DEFAULT',  
        #self.explainer.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.explainer(x)['out']
        return x
# Example usage:
#num_classes = 21  # For Pascal VOC dataset
#model = FCN8s(num_classes)