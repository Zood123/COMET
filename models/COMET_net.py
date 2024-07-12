import torch
torch.cuda.empty_cache()
import torch.nn as nn
from utils import show_image,show_image_one
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
import torch.nn.functional as F
import torchvision
from models.resnet224 import resnet34, resnet18,resnet34_feature
from models.unet_model import UNet,UNetResNet18
from models.FCN import FCN,Deeplabv3Resnet50ExplainerModel


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.4717, 0.4499, 0.3837), 'std': (0.2600, 0.2516, 0.2575),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
    'COMET_net': _cfg(),
}

import time

import timm

class COMET_net(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=9, player_num =1,pretrained = None):
        super().__init__()
        #if pretrained == None:
        print("IS pretrained?: ")
        print(pretrained)
        self.gumbel = False
        self.generator_type = "fcn"
        self.player_num = player_num
        if self.gumbel:
            blockexpansion =1
            self.generator =  resnet34_feature()
            self.mask_head = nn.Linear(512 * blockexpansion,  196)
            self.compl_head = nn.Linear(512 * blockexpansion, 196)
        else:
            if self.generator_type == "unet":
                self.generator =  UNet(3,1)
            elif self.generator_type == "fcn":
                #self.generator = FCN(1) # Deeplabv3Resnet50ExplainerModel(1) #Deeplabv3Resnet50ExplainerModel  UNetResNet18(1)
                self.generator = Deeplabv3Resnet50ExplainerModel(1)
                #self.generator = torchvision.models.segmentation.fcn_resnet50(pretrained=False,num_classes=1)
                #self.generator.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))
            else:
                self.generator =  resnet34(num_classes=196)
        # set up the predictor 
        if self.player_num == 3:
            '''
            self.predictor = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            self.predictor.head = torch.nn.Linear(self.predictor.head.in_features, num_classes)
        
            # Initialize the complement Vision Transformer model (smaller version)
            self.completement_pred = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            self.completement_pred.head = torch.nn.Linear(self.completement_pred.head.in_features, num_classes)
            
            '''
            self.predictor =  torchvision.models.resnet18(pretrained =pretrained)
            self.predictor.fc = torch.nn.Linear(
                in_features=self.predictor.fc.in_features, out_features=num_classes)
            
            self.completement_pred =  torchvision.models.resnet18(pretrained =pretrained)
            self.completement_pred.fc = torch.nn.Linear(
                in_features=self.completement_pred.fc.in_features, out_features=num_classes)
            
            
            '''
            self.predictor =  torchvision.models.mobilenet_v2(pretrained =pretrained)
            self.predictor.classifier[1] = torch.nn.Linear(
                in_features=self.predictor.classifier[1].in_features, out_features=num_classes)
            
            self.completement_pred =  torchvision.models.mobilenet_v2(pretrained =pretrained)
            self.completement_pred.classifier[1] = torch.nn.Linear(
                in_features=self.completement_pred.classifier[1].in_features, out_features=num_classes)
            '''
            #weights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18_pretrained/resnet18-186-best.pth"
            #self.predictor.load_state_dict(torch.load(weights_path))
            #self.completement_pred.load_state_dict(torch.load(weights_path))
            # optinal for fixed com
            
        elif self.player_num == 2:
            self.predictor =  torchvision.models.resnet18(num_classes=num_classes,pretrained =pretrained)
            self.completement_pred =  torchvision.models.resnet18(num_classes=num_classes,pretrained =pretrained)
            weights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
            
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_flower_double/resnet18-16-best.pth"
            
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"
            
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
            self.completement_pred.load_state_dict(torch.load(weights_path))
        else:
            self.predictor =  torchvision.models.resnet18(pretrained = pretrained)
            self.predictor.fc = torch.nn.Linear(
                in_features=self.predictor.fc.in_features, out_features=num_classes)
            weights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_no_fg/resnet18-174-best.pth"
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_only_fg/resnet18-190-best.pth"
            
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"
            #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18_pretrained/resnet18-186-best.pth"
            self.predictor.load_state_dict(torch.load(weights_path))



    def forward_all(self, x):
        #start = time.time()
        input_x = x

        # torch.Size([128, 3, 224, 224])
        # 16*16 patch

        start = time.time()
        if self.gumbel:
            features = self.generator(x)
            mask_score = self.mask_head(features).unsqueeze(-1)
            complement_score = self.compl_head(features).unsqueeze(-1)
            all_score = torch.cat([mask_score, complement_score], dim=2)
            mask = F.gumbel_softmax(all_score, tau=self.tau_gumbel, hard=True, dim=-1) 
        else:
            mask_score =torch.sigmoid(self.generator(x)).unsqueeze(-1)
            # torch.Size([16, 1, 224, 224, 1])
            complement_score= 1-mask_score
            all_score = torch.cat([mask_score, complement_score], dim=-1)
            mask = all_score


        folded_mask_r = mask[:,:,:,:,0]
        folded_mask_c=mask[:,:,:,:,1]
        end = time.time()
        time_used = end-start
        
        #mixed_samples = self.mix_samples(input_x,folded_mask_r)
        #for i,mixed_image in enumerate(mixed_samples):
        #    show_image_one(mixed_image.permute(1, 2, 0).cpu().numpy(),dir="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/mixed_samples/mixed"+str(i)+".png")
        #print()
        #sexit()

        # all_batch = torch.cat((x, mixed_images), dim=0)
        # torch.Size([128, 3, 224, 224])
        if self.player_num ==3 or self.player_num ==2: 
            x = self.predictor(input_x*folded_mask_r)
            #comp = self.predictor(input_x*folded_mask_c)
            comp = self.completement_pred(input_x*folded_mask_c) # com
        else:
            #x = self.predictor(input_x*folded_mask_r)
            #comp = self.predictor(input_x*folded_mask_c)
            q = torch.zeros_like(input_x)
            mean = [0.4717, 0.4499, 0.3837]
            std = [0.2600, 0.2516, 0.2575]
            #q = torch.zeros(size)
            for i in range(3):  # Assuming 3 channels
                q[:, i, :, :] = (0 - mean[i]) / std[i]
            q = q.to(folded_mask_r.device)
            x = self.predictor(input_x*folded_mask_r +  q * (1 - folded_mask_r)) # torch.Size([128, 1, 224, 224])
            # torch.Size([128, 3, 224, 224])
            comp = self.predictor(input_x*folded_mask_c +  q * (1 - folded_mask_c)) 
        

        return x,folded_mask_r,comp #,time_used #was mixed images
    
    def mix_samples(self,images,masks):
        
        num_samples = len(masks)
        mixed_samples = []
        for i in range(num_samples):
            max_diff = 0
            for j in range(num_samples):
                diff = torch.clamp(masks[i] +1 - masks[j],max=1).sum()
                if diff > max_diff and i != j:
                    max_diff = diff
                    max_pair = (i, j)
            mixed_sample = self.soft_mix(images[max_pair[0]],masks[max_pair[0]],images[max_pair[1]],masks[max_pair[1]],0.8)
            mixed_samples.append(mixed_sample)
    
        return torch.stack(mixed_samples)



    def soft_mix(self,image1,mask1,image2,mask2,alpha):
        mixed_image =(image1*mask1*alpha +image2*(1-mask2)*(1-alpha))/(mask1*alpha+(1-mask2)*(1-alpha))
        return mixed_image

    def forward(self, x):
        r_pred,mask,comp = self.forward_all(x)
        
        # 0.01  0.1 for sparsity2
        #+ 0.01*torch.sum( mask- mask*mask)
        return r_pred,comp,mask







@register_model
def COMET_net_builder(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = COMET_net(**kwargs)
    model.default_cfg = default_cfgs['COMET_net']
    return model