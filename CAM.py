import torch 
import torchvision
from torchvision.models import resnet18
import torchcam
from torchcam.methods import CAM
import torch.nn.functional as F
import numpy


# generate the map with the size of 224*224
# output type: tensor
def CAM_mapgeneration(model,input_tensor,target_size=224):
    #input_tensor = torch.zeros((3,224, 224), dtype=torch.float32).unsqueeze(dim=0)
    #print(input_tensor.shape)
    #model = resnet18(pretrained=True).eval() 
    cam = CAM(model, 'layer4', 'fc')
    with torch.no_grad(): 
        out = model(input_tensor)
    _,pred = out.max(1)
    #print(_)
    #print(pred.cpu().item())
    #exit()
    map1 = cam(class_idx=pred.cpu().item())[0]
    #print(map1)
    output_tensor = F.interpolate(map1.unsqueeze(0), size=(target_size, target_size), mode='bilinear',align_corners=False) 
    #print(output_tensor) bilinear False align_corners=False

    return output_tensor

#CAM_mapgeneration()
from torchcam.methods import ScoreCAM
def ScoreCAM_generation(model,input_tensor,target_size=224):

    #model = resnet18(pretrained=True).cuda().eval()
    #input_tensor = torch.zeros((3,224, 224), dtype=torch.float32).unsqueeze(dim=0).cuda()
    cam = ScoreCAM(model, batch_size=1,target_layer ='layer4')
    # with torch.no_grad(): out = model(input_tensor)
    with torch.no_grad(): 
        out = model(input_tensor)
    _,pred = out.max(1)
    map1 = cam(class_idx=pred.cpu().item())[0]
    #print(map1)
    #exit()
    output_tensor = F.interpolate(map1.unsqueeze(0), size=(target_size, target_size), mode='bilinear',align_corners=False) 
    return output_tensor
    #print(map1.shape)
    #exit()