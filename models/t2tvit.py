# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from utils import show_image
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
from models.unet_model import UNet
from models.FCN import FCN


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
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
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        #print("-----------------------see --- x size")
        #print(self.tokens_to_token.num_patches)
        # self.tokens_to_token.num_patches
        # torch.Size([128, 197, 256])

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # torch.Size([128, 197, 256])
        
        x = self.norm(x)
        # torch.Size([128, 197, 256])
        
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x.size(): torch.Size([128, 256])
        
        x = self.head(x)
        return x



    

def unsqueeze_mask(mask):
        mask = mask.unsqueeze(-1)
        #print(mask.shape)
        masked = mask.expand(-1,-1,16*16) # torch.Size([128, 196, 256])
        masked = masked.transpose(1,2)
        # torch.Size([128, 256, 196])
        return masked
 




class T2T_ViT_p2c(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        '''
        self.generator = Rationale_generator(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, depth=3,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, token_dim=token_dim)
        '''
        '''
        self.generator = T2T_ViT(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, num_classes=196,embed_dim=embed_dim, depth=4,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, token_dim=token_dim)
        '''
        self.gumbel = False
        self.generator_type = "fcn"
        if self.gumbel:
            blockexpansion =1
            self.generator =  resnet34_feature()
            self.mask_head = nn.Linear(512 * blockexpansion,  196)
            self.compl_head = nn.Linear(512 * blockexpansion, 196)
        else:
            if self.generator_type == "unet":
                self.generator =  UNet(3,1)
            elif self.generator_type == "fcn":
                self.generator = FCN(1)
                #self.generator = torchvision.models.segmentation.fcn_resnet50(pretrained=False,num_classes=1)
                #self.generator.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))
            else:
                self.generator =  resnet34(num_classes=196)




        #self.predictor  = rationale_predictor(embed_dim,num_classes,num_heads,mlp_ratio,qkv_bias,qk_scale,drop_rate,attn_drop_rate,norm_layer,drop_path_rate,depth=3)
        '''
        self.predictor = T2T_ViT(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, depth=4,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, token_dim=token_dim)
        '''
        self.predictor =  resnet18(num_classes=60)
        self.completement_pred =  resnet18(num_classes=60)
        #self.predictor.load_state_dict(torch.load("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/myresnet_predictor/resnet34-186-best.pth"))



        self.hard_splitx = nn.Unfold(kernel_size=(16, 16), stride=(16, 16))
        self.fold_back  =nn.Fold([224, 224],kernel_size=(16, 16), stride=(16, 16))

        self.proj_x = nn.Linear(768,embed_dim)

        self.tau_gumbel =1

        #self.head_full=  nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, 1) if num_classes > 0 else nn.Identity()
    '''
    def generator_m(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
    '''


    def forward_all(self, x):
        input_x = x

        # torch.Size([128, 3, 224, 224])
        # 16*16 patch

        
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
        # torch.Size([128, 1, 224, 224, 2])
        #print(mask.shape)


        score_matrix =self.matching_img(mask[:,:,:,:,0],mask[:,:,:,:,1])

        #print(score_matrix[0])
        #print(score_matrix[1])

        second_max_indices = np.argpartition(score_matrix, -2, axis=1)[:, -2]

        #print(max_indices)
        #print(second_max_indices)


        #r_masked = unsqueeze_mask(mask_score.squeeze(-1))  # mask[:,:,0]
        #c_masked = unsqueeze_mask(complement_score.squeeze(-1))
        

        
        if self.generator_type == "unet" or "fcn":
            folded_mask_r = mask[:,:,:,:,0]
            folded_mask_c=mask[:,:,:,:,1]
        else:
        
            r_masked = unsqueeze_mask(mask[:,:,0])  # mask[:,:,0]
            c_masked = unsqueeze_mask(mask[:,:,1])
        

            folded_mask_r =self.fold_back(r_masked)
            folded_mask_c =self.fold_back(c_masked)
    
        

        #print(folded_mask_r.shape)
   
        # torch.Size([128, 1, 224, 224])
        input_x_b = input_x[second_max_indices]
        folded_mask_c_mix = folded_mask_c[second_max_indices]
        
        #print((input_x_b.shape))


        mixed_images = self.image_mixing_batch(input_x,folded_mask_r,input_x_b,folded_mask_c_mix)


        x = input_x*folded_mask_r

        # all_batch = torch.cat((x, mixed_images), dim=0)
        all_batch = x
        # torch.Size([128, 3, 224, 224])
        x = self.predictor(all_batch)
        comp = self.completement_pred(input_x*folded_mask_c)
        

        '''
        x = self.hard_splitx(input_x)
        # torch.Size([128, 768, 196])
        x_nomask = x.transpose(1,2)
        x = mask*x_nomask
        # torch.Size([128, 196, 768])
        x = self.predictor(x)
        # 128, 256
        '''
        
        return x,folded_mask_r,mixed_images,comp #was mixed images
    
    def image_mixing_batch(self,images_b1,images_masks1,images_b2,images_masks2):


        mixed_images = []
        
        ##### remember to change
        for i in range(len(images_b1)):
            mixed_image = self.image_softmixing(images_b1[i],images_masks1[i],images_b2[i],images_masks2[i])
            mixed_images.append(mixed_image)

        mixed_images = torch.stack(mixed_images, dim=0)

        return mixed_images




    # union with image token num (196)
    # image 1 priority
    def image_mixing(self,image1,mask1,image2,mask2):
        device = image1.device
        mixed_image = torch.where((mask1 >= 0.5), image1, torch.where((mask2 >= 0.5), image2, torch.tensor(0, dtype=torch.float32).to(device)))

        return mixed_image
    

    def image_softmixing(self,image1,mask1,image2,mask2):
        alpha = 0.8
        mixed_image =(image1*mask1*alpha +image2*mask2*(1-alpha))/(mask1*alpha+mask2*(1-alpha))
        return mixed_image

    # count overlapping
    # if under certain overlapping ratio, then you can mix
    def matching_img(self,target_masks,mix_masks):

        score_matrix = np.zeros((target_masks.shape[0], mix_masks.shape[0]))
        for i,target_mask in enumerate(target_masks):
            for j,mix_mask in enumerate(mix_masks):
                score_matrix[i,j] = self.matching_score(target_mask,mix_mask)

        return score_matrix

    # matching_score
    def matching_score(self,target_mask,mix_mask):
        #print(target_mask)
        #print(mix_mask)
        #exit()
        score = torch.sum(abs(target_mask-mix_mask))

        return score



    #def fore_back_mix (foregouond )
    def background_swap(self,fg_img,fg_mask,bg_imgs,bg_masks):
        mixed_imgs = fg_img.repeat()
                
        return 0

    def forward(self, x):
        r_pred,mask,_,comp = self.forward_all(x)
        
        # 0.01  0.1 for sparsity2
        l1_mask =torch.sum(torch.max(torch.sum(mask,dim=[-1,-2])-244*244*0.2,torch.tensor(0.0))) #+ 0.01*torch.sum( mask- mask*mask)

        return r_pred,comp,l1_mask




class T2T_ViT_Feature(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return x


class T2tvit_Classifier(nn.Module):

    def __init__(self, embed_dim, num_classes=1000, bias=True):
        super(T2tvit_Classifier, self).__init__()
        self.fc =  nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.fc.weight, std=.02)
        if isinstance(self.fc, nn.Linear) and self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return x


@register_model
def T2t_vit_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_4_generator(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=4, num_heads=4, mlp_ratio=2., **kwargs) # 4
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model




@register_model
def T2t_vit_ration(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT_p2c(tokens_type='performer', embed_dim=256, depth=4, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ration_onepred(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = rationresnet_onepred(tokens_type='performer', embed_dim=256, depth=4, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_7_feature(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = T2T_ViT_Feature(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    return model

@register_model
def classifier(pretrained=False, **kwargs): # adopt performer for tokens to token
    classifier_model = T2tvit_Classifier(embed_dim=256, **kwargs)
    return classifier_model


@register_model
def T2t_vit_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_12(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_19(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# rexnext and wide structure
@register_model
def T2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model