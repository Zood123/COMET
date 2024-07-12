from __future__ import absolute_import
import math
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from torch.autograd import Variable
from models.gate_function import virtual_gate
from models.MobileNet import EncoderMobileNetV2
import matplotlib.pyplot as plt
from models.FCN import FCN,Deeplabv3Resnet50ExplainerModel

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, gate_flag=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        if gate_flag is True:
            self.gate = virtual_gate(cfg)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.gate_flag = gate_flag

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate_flag:
            out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2, 2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([x.data, zero_pads], dim=1))
    return out


class ResNet(nn.Module):
    def __init__(self, depth, dataset='cifar10', cfg=None, width=1, gate_flag=False):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[round(width * 16)] * n, [round(width * 32)] * n, [round(width * 64)] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = round(width * 16)
        self.conv1 = nn.Conv2d(3, round(width * 16), kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(width * 16))
        self.relu = nn.ReLU(inplace=True)
        self.gate_flag = gate_flag
        self.base_width = width * 16

        self.layer1 = self._make_layer(block, round(width * 16), n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, round(width * 32), n, cfg=cfg[n:2 * n], stride=2)
        self.layer3 = self._make_layer(block, round(width * 64), n, cfg=cfg[2 * n:3 * n], stride=2)
        self.avgpool = nn.AvgPool2d(56) # 8 for cifar
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == "Imagenet9":
            num_classes =9
        self.fc = nn.Linear(round(width * 64) * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, gate_flag=self.gate_flag))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i], gate_flag=self.gate_flag))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape) # torch.Size([128, 3, 224, 224])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32 x 32

        x = self.layer1(x)  # 32 x 32
        x = self.layer2(x)  # 16 x 16
        x = self.layer3(x)  # 8 x 8
        #print(x.shape) # torch.Size([128, 64, 56, 56])
        x = self.avgpool(x)
        #print(x.shape) # torch.Size([128, 64, 7, 7])
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #print("------------------------")
        x = self.fc(x)

        return x

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end
                i += 1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                all_grad.append(m.get_grads().clone())
        return all_grad

    def foreze_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            elif isinstance(m, nn.Conv2d):
                m.eval()
                m.weight.requires_grad = False
            elif isinstance(m, nn.Linear):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.eval()
                count += 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


class EncoderResNet(nn.Module):
    def __init__(self, depth, dataset='cifar10', cfg=None, width=1, gate_flag=False):
        super(EncoderResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[round(width * 16)] * n, [round(width * 32)] * n, [round(width * 64)] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = round(width * 16)
        self.conv1 = nn.Conv2d(3, round(width * 16), kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(width * 16))
        self.relu = nn.ReLU(inplace=True)
        self.gate_flag = gate_flag
        self.base_width = width * 16

        self.layer1 = self._make_layer(block, round(width * 16), n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, round(width * 32), n, cfg=cfg[n:2 * n], stride=2)
        self.layer3 = self._make_layer(block, round(width * 64), n, cfg=cfg[2 * n:3 * n], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise NotImplementedError
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, gate_flag=self.gate_flag))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i], gate_flag=self.gate_flag))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)

        return scale1, scale2, scale3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('*' * 20)
        logging.warning('Encoder ResNet:')
        logging.warning('loading pretrained checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        if 'module' in list(checkpoint['model'].keys())[0]:
            new_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if 'fc' in k:
                    continue
                name = k.replace('module.', '')  # remove `module.`
                if 'resnet' in name:
                    name = name.replace('resnet.', '')
                elif 'model' in name:
                    name = name.replace('resnet.', '')
                new_dict[name] = v
            self.load_state_dict(new_dict)
        else:
            for k in list(checkpoint['model'].keys()):
                if 'linear' in k:
                    del checkpoint['model'][k]
            if 'resnet' in list(checkpoint['model'].keys())[0]:
                new_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k.replace('resnet.', '')
                    new_dict[name] = v
            self.load_state_dict(checkpoint['model'])


class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def CNNBlock(in_channels, out_channels,
             kernel_size=3, layers=1, stride=1,
             follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):
    assert layers > 0 and kernel_size % 2 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer == 0 else 1,
                                  padding=int(kernel_size / 2), bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return nn.Sequential(*_modules)


def SubpixelUpsampler(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False),
                      follow_with_bn=True):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels, out_channels=out_channels)
        self.follow_up = Block(out_channels + passthrough_channels, out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out, passthrough), 1)
        return self.follow_up(out)


class RealTimeSaliencyModel(nn.Module):
    def __init__(self, args):
        super(RealTimeSaliencyModel, self).__init__()
        self.args = args

        if self.args['model_type'] == 'resnet-gate' or self.args['model_type'] == 'original-resnet-gate':
            #self.resnet = EncoderResNet(self.args.depth, gate_flag=True)
            #self.uplayer3 = UpSampleBlock(in_channels=64, out_channels=32, passthrough_channels=32)
            #self.uplayer2 = UpSampleBlock(in_channels=32, out_channels=16, passthrough_channels=16)
            #self.embedding = nn.Embedding(self.args.num_classes, 64)
            self.generator =  Deeplabv3Resnet50ExplainerModel(16)
            self.saliency_chans = nn.Conv2d(16, 2, kernel_size=1, bias=True)#32

        elif self.args.model_type == 'MobileNetV2':
            self.resnet = EncoderMobileNetV2()
            self.uplayer3 = UpSampleBlock(in_channels=1280, out_channels=96, passthrough_channels=96)
            self.uplayer2 = UpSampleBlock(in_channels=96, out_channels=32, passthrough_channels=32)
            self.embedding = nn.Embedding(self.args.num_classes, 1280)
            self.saliency_chans = nn.Conv2d(32, 2, kernel_size=1, bias=False)

        self.fix_encoder = self.args['fix_encoder']

        #if self.args.load_saliency_encoder_from_checkpoint:
        #    self.resnet.load_checkpoint(self.args.pretrained_model_dir)

    def forward(self, x, labels):
        '''
        if self.fix_encoder:
            self.resnet.eval()
            with torch.no_grad():
                scale1, scale2, scale3 = self.resnet(x)
        else:
            scale1, scale2, scale3 = self.resnet(x)

        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale3 * em.view(-1, em.shape[1], 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale3 = scale3 * th

        upsample2 = self.uplayer3(scale3, scale2)
        upsample1 = self.uplayer2(upsample2, scale1)
        '''
        upsample1 = self.generator(x)
        saliency_chans = self.saliency_chans(upsample1)
        #print(saliency_chans)
        #print(saliency_chans.shape)
        #exit()
        a = torch.abs(saliency_chans[:, 0, :, :])
        b = torch.abs(saliency_chans[:, 1, :, :])
        return torch.unsqueeze(a / (a + b + 1e-8), dim=1)

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('loading pretrained checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if 's_model' in k:
                name = k.replace('s_model.', '')
                if 'module' in name:
                    name = name.replace('module.', '')
                new_dict[name] = v
            elif 'selector' in k:
                name = k.replace('selector.', '')
                if 'module' in name:
                    name = name.replace('module.', '')
                new_dict[name] = v

        self.load_state_dict(new_dict)


class RealTimeSaliencyRBF(nn.Module):
    def __init__(self, args):
        super(RealTimeSaliencyRBF, self).__init__()
        self.args = args
        if self.args['model_type'] == 'resnet-gate' or self.args['model_type'] == 'original-resnet-gate':
            #self.resnet = EncoderResNet(self.args['depth'], gate_flag=True)
            #self.uplayer3 = UpSampleBlock(in_channels=64, out_channels=32, passthrough_channels=32)
            #self.uplayer2 = UpSampleBlock(in_channels=32, out_channels=16, passthrough_channels=16)
            #self.embedding = nn.Embedding(self.args['class_num'], 64)
            self.generator =  Deeplabv3Resnet50ExplainerModel(16)
            self.saliency_chans = nn.Conv2d(16, 3, kernel_size=224, bias=True)#32

        elif self.args['model_type']  == 'MobileNetV2':
            #self.resnet = EncoderMobileNetV2()
            #self.uplayer3 = UpSampleBlock(in_channels=1280, out_channels=96, passthrough_channels=96)
            #self.uplayer2 = UpSampleBlock(in_channels=96, out_channels=32, passthrough_channels=32)
            #self.embedding = nn.Embedding(self.args['class_num'], 1280)
            self.generator =  Deeplabv3Resnet50ExplainerModel(16)
            self.saliency_chans = nn.Conv2d(32, 3, kernel_size=224, bias=True)#32

        self.fix_encoder = self.args['fix_encoder']
        #if self.args.load_saliency_encoder_from_checkpoint:
        #    self.resnet.load_checkpoint(self.args.pretrained_model_dir)

        if self.args['initial_sigma']:
            self.saliency_chans.bias.data[2].fill_(self.args['initial_sigma'])

    def forward(self, x, labels):
        '''
        if self.fix_encoder:
            self.resnet.eval()
            with torch.no_grad():
                scale1, scale2, scale3 = self.resnet(x)
        else:
            scale1, scale2, scale3 = self.resnet(x)

        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale3 * em.view(-1, em.shape[1], 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale3 = scale3 * th

        upsample2 = self.uplayer3(scale3, scale2)
        upsample1 = self.uplayer2(upsample2, scale1)
        '''
        upsample1 = self.generator(x)
        #print(upsample1.shape)
        #torch.Size([128, 16, 224, 224])
        saliency_params = self.saliency_chans(upsample1)
        #print("--------output__-coordinate")
        #print(saliency_params)
        #torch.Size([128, 3, 193, 193])   # modified torch.Size([128, 3, 1, 1])
        #print(saliency_params.shape)
        #exit()
        masks = self.calculate_rbf(saliency_params)
        #print(masks.shape)
        #exit()
        return masks, saliency_params

    def calculate_rbf(self, saliency_params):
        
        params = saliency_params.squeeze()
        
        if len(params.shape) == 1:
            params = params.unsqueeze(0)  # 
        # added------------------------------------------
        params[:, 2] =torch.clamp(params[:, 2], 10, 100) 
        # ------------------------ [-108,108]->[4,220]
        xy = (108. * torch.tanh(params[:, :2] / 108) + 112.).cuda()  # we use tanh relative to the center of image and add 16 to make
        # coordinates relative to top-left. (the final values will be in the center 28*28 frame of the image)
        sigma = (torch.logaddexp(torch.zeros_like(params[:, 2]), params[:, 2]) + 1e-8).cuda()  # sigma = log(1 + exp(m_x)) variance
        maps = []
        for i in range(params.shape[0]):
            x_c, y_c = self.coordinate_arrays()
            map = (1 / (2 * torch.tensor(np.pi) * (sigma[i] ** 2))) * \
                  torch.exp((-1. / (2 * (sigma[i] ** 2))) * (((x_c - xy[i, 0]) ** 2) + ((y_c - xy[i, 1]) ** 2)))  # Calculating RBF kernel at each pixel.
            new_map = map.unsqueeze(0) / (map.detach().max() + 1e-8)  # Converting gaussian density to RBF.
            maps.append(new_map)

        out_maps = (torch.stack(maps)).cuda()
        return out_maps

    def coordinate_arrays(self):
        y_coordinates = ((torch.arange(224.)).repeat((224, 1))).cuda()
        x_coordinates = (torch.transpose((torch.arange(224.)).repeat((224, 1)), 1, 0)).cuda()
        return x_coordinates, y_coordinates

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('#' * 20)
        logging.warning('loading pretrained selector from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if 'selector' in k:
                name = k.replace('selector.', '')
                if 'module' in name:
                    name = name.replace('module.', '')
                new_dict[name] = v

        self.load_state_dict(new_dict)