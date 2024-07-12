import torch
from bcos.modules.bcosconv2d import BcosConv2d
import torchvision

import hubconf











if __name__ =="__main__":
    model = hubconf.resnet18(pretrained=False)
    #print(model)
    num_class = 9
    model[0].fc = BcosConv2d(
                    in_channels=model[0].fc.in_channels, out_channels=num_class)
    
    print(model)
    
