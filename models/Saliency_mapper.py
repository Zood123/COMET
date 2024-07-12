import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch.autograd import Variable
import numpy as np

class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)
      

def CNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):

        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=int(kernel_size/2), bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)

def SubpixelUpsampler(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)

class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels,out_channels=out_channels)
        self.follow_up = Block(out_channels+passthrough_channels,out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out,passthrough), 1)
        return self.follow_up(out)



class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out







class SaliencyModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10): # 10 for imagenet and 60 for nico
        super(SaliencyModel, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=num_blocks[3], stride=2)
        

        self.uplayer4 = UpSampleBlock(in_channels=512,out_channels=256,passthrough_channels=256)
        self.uplayer3 = UpSampleBlock(in_channels=256,out_channels=128,passthrough_channels=128)
        self.uplayer2 = UpSampleBlock(in_channels=128,out_channels=64,passthrough_channels=64)
        
        self.embedding = nn.Embedding(num_classes,512)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.saliency_chans = nn.Conv2d(64,2,kernel_size=1,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    
    def forward(self, x,labels):
        out = F.relu(self.bn1(self.conv1(x)))
        
        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)
        scale4 = self.layer4(scale3)

      
        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4*em.view(-1, 512, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4*th
        
        
        upsample3 = self.uplayer4(scale4,scale3)
        upsample2 = self.uplayer3(upsample3,scale2)
        upsample1 = self.uplayer2(upsample2,scale1)
        
        saliency_chans = self.saliency_chans(upsample1)
        
        
        #out = F.avg_pool2d(scale4, 4)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        
        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        
        return torch.unsqueeze(a/(a+b), dim=1) #, out


def saliency_model():
    return SaliencyModel(Block, [2,2,2,2])




class Posthoc_loss: 
    
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.area_loss_coef = 8 # paper 10 , 8
        self.smoothness_loss_coef = 0.5 # paper 1e-3   0.5
        self.preserver_loss_coef = 0.3 # paper1  0.3
        self.destroyer_loss_coef = 1 # paper 5,  1
        self.destroyer_loss_power = 1 # paper 0.3,  1
        self.area_loss_power =0.3 # paper 1 ,  0.3 
     
    def get(self, masks, images, targets, black_box_func):
    
        one_hot_targets = self.one_hot(targets)
        
        area_loss = self.area_loss(masks)
        smoothness_loss = self.smoothness_loss(masks)

        destroyer_loss = self.destroyer_loss(images,masks,one_hot_targets,black_box_func)
        preserver_loss = self.preserver_loss(images,masks,one_hot_targets,black_box_func)
        
        
        return self.destroyer_loss_coef*(destroyer_loss)**(self.destroyer_loss_power) + self.area_loss_coef*area_loss + self.smoothness_loss_coef*smoothness_loss + self.preserver_loss_coef*preserver_loss
        
    def one_hot(self,targets):
        depth = self.num_classes
        if targets.is_cuda:
            return Variable(torch.zeros(targets.size(0), depth).cuda().scatter_(1, targets.long().view(-1, 1).data, 1))
        else:
            return Variable(torch.zeros(targets.size(0), depth).scatter_(1, targets.long().view(-1, 1).data, 1))

  
    def tensor_like(self,x):
        if x.is_cuda:
            return torch.Tensor(*x.size()).cuda()
        else:
            return torch.Tensor(*x.size())
  
    def area_loss(self, masks):
        if self.area_loss_power != 1:
            masks = (masks+0.0005)**self.area_loss_power # prevent nan (derivative of sqrt at 0 is inf)

        return torch.mean(masks)
  
    def smoothness_loss(self,masks, power=2, border_penalty=0.3):
        x_loss = torch.sum((torch.abs(masks[:,:,1:,:] - masks[:,:,:-1,:]))**power)
        y_loss = torch.sum((torch.abs(masks[:,:,:,1:] - masks[:,:,:,:-1]))**power)
        if border_penalty>0:
            border = float(border_penalty)*torch.sum(masks[:,:,-1,:]**power + masks[:,:,0,:]**power + masks[:,:,:,-1]**power + masks[:,:,:,0]**power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * masks.size(0))  # watch out, normalised by the batch size!
  
    def destroyer_loss(self,images,masks,targets,black_box_func):

        destroyed_images = self.apply_mask(images,1 - masks)
        out = black_box_func(destroyed_images.cuda())
        
        return self.cw_loss(out, targets, targeted=False, t_conf=1., nt_conf=5)
  
    def preserver_loss(self,images,masks,targets,black_box_func):
        preserved_images = self.apply_mask(images,masks)
        out = black_box_func(preserved_images)
        
        return self.cw_loss(out, targets, targeted=True, t_conf=1., nt_conf=1)
  
    def apply_mask(self,images, mask, noise=False, random_colors=True, blurred_version_prob=0.5, noise_std=0.11,
                 color_range=0.66, blur_kernel_size=55, blur_sigma=11,
                 bypass=0., boolean=False, preserved_imgs_noise_std=0.03):
        images = images.clone()
        cuda = images.is_cuda

        if boolean:
            # remember its just for validation!
            return (mask > 0.5).float() *images

        assert 0. <= bypass < 0.9
        n, c, _, _ = images.size()

        if preserved_imgs_noise_std > 0:
            images = images + Variable(self.tensor_like(images).normal_(std=preserved_imgs_noise_std) , requires_grad=False)
        if bypass > 0:
            mask = (1.-bypass)*mask + bypass
        if noise and noise_std:
            alt = self.tensor_like(images).normal_(std=noise_std)
        else:
            alt = self.tensor_like(images).zero_()
        if random_colors:
            if cuda:
                alt += torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range/2., color_range/2.)
            else:
                alt += torch.Tensor(n, c, 1, 1).uniform_(-color_range/2., color_range/2.)

        alt = Variable(alt, requires_grad=False)

        if blurred_version_prob > 0.: # <- it can be a scalar between 0 and 1
            cand = self.gaussian_blur(images, kernel_size=blur_kernel_size, sigma=blur_sigma)
            if cuda:
                when = Variable((torch.Tensor(n, 1, 1, 1).cuda().uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
            else:
                when = Variable((torch.Tensor(n, 1, 1, 1).uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
            alt = alt*(1.-when) + cand*when

        return (mask*images.detach()) + (1. - mask)*alt.detach()

    def cw_loss(self,logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):

        this = torch.sum(logits*one_hot_labels, 1)
        other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        t = F.relu(other_best - this + t_conf)
        nt = F.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)

    def gaussian_blur(self,_images, kernel_size=55, sigma=11):
        ''' Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
        Returns blurred images of the same size. Kernel size must be odd.
        Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.'''
        
        kernel_a, kernel_b = self._gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
        kernel_a = torch.Tensor(kernel_a)
        kernel_b = torch.Tensor(kernel_b)
        if _images.is_cuda:
            kernel_a = kernel_a.cuda()
            kernel_b = kernel_b.cuda()
        _rows = conv2d(_images, Variable(kernel_a, requires_grad=False), groups=_images.size(1), padding=(int(kernel_size / 2), 0))
        return conv2d(_rows, Variable(kernel_b, requires_grad=False), groups=_images.size(1), padding=(0, int(kernel_size / 2)) )


    def _gaussian_kernels(self,kernel_size, sigma, chans):
        assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
        x = np.expand_dims(np.array(range(int(-kernel_size/2), int(-kernel_size/2)+kernel_size, 1)), 0)
        vals = np.exp(-np.square(x)/(2.*sigma**2))
        _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
        kernel =  np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
        return kernel, np.transpose(kernel, [0, 1, 3, 2])

   