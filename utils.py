import os
import sys
import re
import datetime
import pickle
import numpy as np
import json
import torch
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim, autograd
import matplotlib.pyplot as plt
from PIL import Image
import pickle







def draw_pr(precision,recall,thresholds):
    # Example precision and recall values for 20 thresholds
    # These would be replaced by the actual values from your data
    #precision = np.linspace(0.9, 0.1, 20)
    #recall = np.linspace(0.1, 0.9, 20)
    #thresholds = np.linspace(0, 1, 20)
    print(thresholds)
    thresholds  = thresholds[::-1]
    thresholds = np.insert(thresholds, 0,1 )
    print(thresholds)
    print(recall)
    print(precision)
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, marker='o')
    plt.ylim(0, 1)
    # Annotate each point with its corresponding threshold value
    #for i, threshold in enumerate(thresholds):
    #    plt.annotate(f'{threshold:.2f}', (recall[i], precision[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Thresholds')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/Draw_graph/precision_recall_curve.png')
    plt.close()



def show_image(img0,img1,img2,dir,std = [0.21851876, 0.2175944, 0.22552039],mean = [0.52418953, 0.5233741, 0.44896784]):
    if not os.path.exists(dir):
        os.mkdir(dir)   

    mean_channel = np.array([mean[0], mean[1], mean[2]])
    std_channel = np.array([std[0], std[1], std[2]])
    img0 = img0 * std_channel + mean_channel
    img2 = img2 * std_channel + mean_channel


    img0 = np.clip(img0, 0, 1)
    img2 = np.clip(img2, 0, 1)

    #print(img1.shape)
    plt.imsave(dir+'/masked_image.png', img0)
    plt.imsave(dir+'/mask.png', img1)
    plt.imsave(dir+'/image.png', img2)
    #plt.imsave(dir+'/mixed_image.png', mixed_img)

    return 1

def show_image_one(img0,std = [0.21851876, 0.2175944, 0.22552039],mean = [0.52418953, 0.5233741, 0.44896784],dir=None):
    mean_channel = np.array([mean[0], mean[1], mean[2]])
    std_channel = np.array([std[0], std[1], std[2]])
    img0 = img0 * std_channel + mean_channel
    img0 = np.clip(img0, 0, 1)
    plt.imsave(dir, img0)

    return 0

def upscale_mask(mask):
    target_size = 224
    scaled_mask = np.zeros([target_size,target_size])
    width_scale=height_scale=target_size // mask.shape[0]
    for i in range(target_size):
        for j in range(target_size):
            scaled_mask[i, j] = mask[i // width_scale, j // height_scale]

    return scaled_mask



def get_network(args,config):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == "posthoc_masker":
        from models.Saliency_mapper import saliency_model
        net = saliency_model()
    elif args.net ==  'Bcos_resnet18':
        import hubconf
        import bcos
        net = hubconf.resnet18(pretrained=False)
        net[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
                    in_channels=net[0].fc.in_channels, out_channels=config['class_num'])
    elif args.net == "RBF_trainpredictor":
        from models.models_rbf import TrainPredictor
        net = TrainPredictor(config)
    elif args.net == "RBF_trainselector":
        from models.models_rbf import TrainSelector
        net = TrainSelector(config)
    elif args.net == 'resnet18':
        net = torchvision.models.resnet18(pretrained=config['pretrained'])
        net.fc = torch.nn.Linear(
                in_features=net.fc.in_features, out_features=config['class_num'])

    elif args.net == 'resnet34':
        net = torchvision.models.resnet34(pretrained=config['pretrained'])
        net.fc = torch.nn.Linear(
                in_features=net.fc.in_features, out_features=config['class_num'])
    elif args.net == 'resnet50':
        net = torchvision.models.resnet50(pretrained=config['pretrained'])
        net.fc = torch.nn.Linear(
                in_features=net.fc.in_features, out_features=config['class_num'])
    elif args.net == 'resnet101':
        net = torchvision.models.resnet101(pretrained=config['pretrained'])
        net.fc = torch.nn.Linear(
                in_features=net.fc.in_features, out_features=config['class_num'])
    elif args.net == 'resnet152':
        net = torchvision.models.resnet152(pretrained=config['pretrained'])
        net.fc = torch.nn.Linear(
                in_features=net.fc.in_features, out_features=config['class_num'])
    elif args.net == 'resnet18cbam':
        from models.resnet_cbam import resnet18_cbam
        net = resnet18_cbam(num_classes=config['class_num'])
    elif args.net == 'resnet18cbam2':
        from models.resnet_cbam2 import ResidualNet
        net = ResidualNet("ImageNet", 18, num_classes=config['class_num'], att_type='CBAM')
    elif args.net == 'resnet18senet':
        from models.resnet_senet import se_resnet18
        net = se_resnet18(num_classes=config['class_num'])
    elif args.net == 'resnet34cbam2':
        from models.resnet_cbam2 import ResidualNet
        net = ResidualNet("ImageNet", 34, num_classes=config['class_num'], att_type='CBAM')
    elif args.net == 'resnet18_ours_cbam':
        from models.resnet_ours_cbam import ResidualNet
        net = ResidualNet("ImageNet", 18, num_classes=config['class_num'], att_type='CBAM')
    elif args.net == 'resvit18':
        from models.resvit import resvit18
        net = resvit18(num_classes=config['class_num'])
    elif args.net == 't2tvit7':
        from models.t2tvit import T2t_vit_4_generator
        net = T2t_vit_4_generator(num_classes=config['class_num'])
    elif args.net == 'COMET_net':
        from models.COMET_net import COMET_net
        net = COMET_net(num_classes=config['class_num'],player_num=config["player_num"],pretrained=config["pretrained"]) 
    elif args.net == 't2tvit12':
        from models.t2tvit import T2t_vit_12
        net = T2t_vit_12(num_classes=config['class_num'])
    #<------------------------------------------------>
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()


    return net


def get_custom_network(args, variance_opt):
    if args.net == 'resnet18':
        from models.resnet224 import resnet18_feature, classifier
    elif args.net == 'resnet18cbam2':
        from models.resnet_cbam2 import resnet18_feature, classifier
    elif args.net == 'resnet18_ours_cbam':
        from models.resnet_ours_cbam import ResidualNet, classifier
    elif args.net == 'resnet18_ours_cbam_multi':
        from models.resnet_ours_cbam_multi import ResidualNet, classifier
    elif args.net == 't2tvit7':
        from models.t2tvit import T2t_vit_7_feature, classifier
    elif args.net == 't2tvit7_ours':
        from models.t2tvit_ours import T2t_vit_7_feature, classifier


    if variance_opt['mode'] in ['ours']:
        num_env = variance_opt['n_env']
        model_list = []
        for e in range(num_env + 1):
            if (e <= num_env - 1):
                model_list.append(classifier(num_classes=60, bias=False))
            else:
                if variance_opt['erm_flag']:
                    model_list.append(classifier(num_classes=60))
                if args.net == 't2tvit7_ours':
                    model_list.append(T2t_vit_7_feature(num_classes=60, final_k=variance_opt['final_k']))
                else:
                    try:
                        model_list.append(ResidualNet("ImageNet", 18, num_classes=60, att_type='CBAM', split_layer=variance_opt['split_layer']))
                    except:
                        model_list.append(ResidualNet("ImageNet", 18, num_classes=60, att_type='CBAM'))

    if args.gpu: #use_gpu
        model_list = [model_list_.cuda() for model_list_ in model_list]
    if args.multigpu:
        model_list = [nn.DataParallel(model_list_) for model_list_ in model_list]

    return model_list


def get_custom_network_vit(args, variance_opt):
    if args.net == 'resvit18':
        from models.resvit import resvit18_feature, classifier

    if variance_opt['mode'] in ['ours']:
        num_env = variance_opt['n_env']
        model_list = []
        for e in range(num_env + 1):
            if (e <= num_env - 1):
                model_list.append(classifier(num_classes=2, bias=False))
            else:
                if variance_opt['erm_flag']:
                    model_list.append(classifier(num_classes=2))
                model_list.append(ResidualNet("ImageNet", 18, num_classes=2, att_type='CBAM'))

    if args.gpu: #use_gpu
        model_list = [model_list_.cuda() for model_list_ in model_list]

    return model_list

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)
    return config

def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv

def get_mean_std(image_folder):
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor
    dataset = ImageFolder(root=image_folder, transform=ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=8)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data, _ in dataloader:
        for dim in range(3):
            mean[dim] += data[:, dim, :, :].mean()
            std[dim] += data[:, dim, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return list(mean.numpy()), list(std.numpy())


def reformulate_data_dist(datas, labels, contexts, config):
    balance_factor = config['variance_opt']['balance_factor']
    training_dist = config['variance_opt']['training_dist']
    cxt_dic = json.load(open(config['cxt_dic_path'], 'r'))
    class_dic = json.load(open(config['class_dic_path'], 'r'))

    new_data = []
    new_label = []
    new_context = []

    for img_class in training_dist.keys():
        cls_num = len(training_dist[img_class])
        class_idx = np.where(np.array(labels) == str(class_dic[img_class]))[0]
        img_class_labels = [labels[idx] for idx in class_idx]
        img_class_datas = [datas[idx] for idx in class_idx]
        img_class_contexts = [contexts[idx] for idx in class_idx]

        for index, img_context in enumerate(training_dist[img_class]):
            img_context_label = cxt_dic[img_context]
            idx = np.where(np.array(img_class_contexts) == str(img_context_label))[0]
            img_context_num = idx.shape[0]
            select_context_num = int(img_context_num * (balance_factor**(index / (cls_num - 1.0))))
            np.random.shuffle(idx)

            selec_idx = idx[:select_context_num]
            new_data.extend([img_class_datas[i] for i in selec_idx])
            new_label.extend([img_class_labels[i] for i in selec_idx])
            new_context.extend([img_class_contexts[i] for i in selec_idx])

    return new_data, new_label, new_context


def make_env(image, label, context, n_env, env_type, pre_split=None):
    # divid the data into `n_env` environments according to `env_type`

    sample_num = len(image)
    sample_env = sample_num // n_env
    image_env = []
    label_env = []
    context_env = []

    if env_type == 'semi-auto':
        # sort according to context
        sort_zip = sorted(zip(image, label, context), key=lambda x:x[2])
        image, label, context = [list(x) for x in zip(*sort_zip)]
        for env_idx in range(n_env):
            start_idx = env_idx * sample_env
            end_idx = (env_idx+1) * sample_env if env_idx != n_env-1 else sample_num
            image_env.append(image[start_idx:end_idx])
            label_env.append(label[start_idx:end_idx])
            context_env.append(context[start_idx:end_idx])

    elif env_type == 'random':
        import random
        sort_zip = list(zip(image, label, context))
        random.shuffle(sort_zip)
        image, label, context = [list(x) for x in zip(*sort_zip)]
        for env_idx in range(n_env):
            start_idx = env_idx * sample_env
            end_idx = (env_idx+1) * sample_env if env_idx != n_env-1 else sample_num
            image_env.append(image[start_idx:end_idx])
            label_env.append(label[start_idx:end_idx])
            context_env.append(context[start_idx:end_idx])

    elif env_type in ['auto-baseline', 'auto-iter', 'auto-iter-cluster']:
        ## use a reference model to make the env split
        ### initialize a split distribution
        assert pre_split is not None
        num_env = pre_split.size(1)
        sort_zip = list(zip(image, label, context))
        for idx in range(num_env):
            sort_zip_idx = [sort_zip[k] for k in torch.where(pre_split[:,idx]==1)[0]]
            image_, label_, context_ = [list(x) for x in zip(*sort_zip_idx)]
            image_env.append(image_)
            label_env.append(label_)
            context_env.append(context_)

    return image_env, label_env, context_env

import random
def load_Imagenet9(dataroot, balance_factor=1, config=None,Data_version="im_original"):
    random.seed(1)
    if Data_version=="im_original":
        file = open("/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_"+dataroot.split("/")[-1]+".pickle", 'rb') #mixed_next; original ; only_fg; no_fg;only_bg_b;only_bg_t
    elif  Data_version=="im_mixed_next":
        file = open("/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_mixed_next_"+dataroot.split("/")[-1]+".pickle", 'rb')


    context_list = ['rock', 'dim', 'water', 'autumn', 'outdoor', 'grass']
    data_addresses= pickle.load(file)
    file.close()
    #print(len(data_addresses))

    #exit()
    #all_file_name = os.listdir(dataroot)
    all_file_name = data_addresses
    #print(len(all_file_name[0]))
    #print(all_file_name[0])
    
    all_data = []
    all_label = []
    all_context = []
    for i,category in enumerate(all_file_name):
        for file_name in category:
            label = i
            #context = context_list.index(file_name.split("/")[-3])
            context = 'rock'
            all_label.append(label)
            all_context.append(context)
            all_data.append(Image.open(file_name).convert('RGB'))
    
    #label_set = list(set(all_label))
    #label2train = {label_set[i]: i for i in range(len(label_set))}

    shuffle = False
    if shuffle==True:
        #print(all_label)
        shuffled_labels = all_label[:]
        random.shuffle(shuffled_labels)
        return all_data, shuffled_labels, all_context
    
    if balance_factor != 1:
        all_data, all_label, all_context = reformulate_data_dist(all_data,all_label,all_context,config)


    print(dataroot.split("/")[-1])
    print(len(all_label))
    print(len(all_data))
    print(len(all_context))
    #exit()

    return all_data, all_label, all_context






def load_NICO(dataroot, balance_factor=1, config=None):
    random.seed(1)
    file = open("/home/xzz5508/code/Imbalance_ood/NICO_dataset/NICO_comman_longtail/NICO_balancedcontext0_"+dataroot.split("/")[-1]+".pickle", 'rb')

    
    context_list = ['rock', 'dim', 'water', 'autumn', 'outdoor', 'grass']
    data_addresses= pickle.load(file)
    file.close()
    #print(len(data_addresses))

    #exit()
    #all_file_name = os.listdir(dataroot)
    all_file_name = data_addresses
    all_data = []
    all_label = []
    all_context = []
    for i,category in enumerate(all_file_name):
        for file_name in category:
            label = i
            context = context_list.index(file_name.split("/")[-3])
            all_label.append(label)
            all_context.append(context)
            all_data.append(Image.open(file_name).convert('RGB'))
    
    #label_set = list(set(all_label))
    #label2train = {label_set[i]: i for i in range(len(label_set))}


    shuffle = True
    if shuffle==True:
        #print(all_label)
        shuffled_labels = all_label[:]
        random.shuffle(shuffled_labels)
        return all_data, shuffled_labels, all_context
        #print(shuffled_labels)
        #exit()

    if balance_factor != 1:
        all_data, all_label, all_context = reformulate_data_dist(all_data,all_label,all_context,config)
        
    #print(dataroot.split("/")[-1])
    #print(len(all_label))
    #print(len(all_data))
    #print(len(all_context))
    #exit()
    return all_data, all_label, all_context


class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label, all_context, transform=None, require_context=False, soft_split=None, label2train=None):
        super(NICO_dataset, self).__init__()
        self.all_data =  all_data
        self.all_label = all_label
        self.all_context = all_context
        self.transform = transform
        self.require_context = require_context

        if label2train is None:
            label_set = list(set(self.all_label))
            label_set.sort()
            self.label2train = {label_set[i]:i for i in range(len(label_set))}
        else:
            self.label2train = label2train

        if soft_split is not None:
            self.soft_split = soft_split
        else:
            self.soft_split = None


    def __getitem__(self, item):
        img = self.all_data[item]
        img = self.transform(img)

        label = self.label2train[self.all_label[item]]
        context = self.all_context[item]

        if self.require_context:
            return img, label, context

        if self.soft_split is not None:
            return img, label, item

        return img, label


    def __len__(self):
        return len(self.all_data)




# you want top 20%
# the quantile should be 0.8 to find threshold
# you want top 20% （0.2 ,false)
def binarize_map(mask, percentage=0.2,complement=False):
    # Convert NumPy array to PyTorch tensor
    #mask = torch.tensor(mask)


    # Calculate the threshold using PyTorch functions
    if not complement:
        threshold = torch.quantile(mask.flatten(),1-percentage)
        binarized_mask = torch.where(mask >= threshold, 1.0, 0.0)
        return binarized_mask

    else:
        threshold = torch.quantile(mask.flatten(),1-percentage)
        binarized_mask = torch.where(mask < threshold, 1.0, 0.0)
        #print(threshold)
        return binarized_mask



# Example usage:
# Assuming your input mask is a NumPy array
import numpy as np


def binarize_map_np(mask,percentage=20):

    #mask = np.array(mask)
    #mask = mask / np.max(mask)
    # Binarize the mask
    flat_mask = mask.flatten()
    threshold = np.percentile(flat_mask, 100 - percentage)

    binarized_mask = (mask > threshold).astype(np.float32)
    
    #binarized_mask = np.clip(mask, epsilon, 1.0 - epsilon)


    return binarized_mask

def load_cifar10(dataset):
    all_context = None
    all_data = [[] for i in range(10)]

    count = 0
    for i,(image, target)in enumerate(dataset):
        all_data[target].append(image)
        #all_label.append(target)

    return all_data


def cifar_train(dataset):
    train_data = [[] for  i in range(10)]
    for i in range(10):
        train_data[i] = dataset[i][1000:]

    all_data =  []
    all_label = []
    all_context = []
    for i, cls_images in enumerate(train_data):
        for image in cls_images:
            all_data.append(image)
            all_label.append(i)
            all_context.append(0)
    return all_data,all_label,all_context




def cifar_val(dataset):
    train_data = [[] for  i in range(10)]
    for i in range(10):
        train_data[i] = dataset[i][:1000]

    all_data =  []
    all_label = []
    all_context = []
    for i, cls_images in enumerate(train_data):
        for image in cls_images:
            all_data.append(image)
            all_label.append(i)
            all_context.append(0)
    return all_data,all_label,all_context



def cifar_test(dataset):


    all_data =  []
    all_label = []
    all_context = []
    for i, cls_images in enumerate(dataset):
        for image in cls_images:
            all_data.append(image)
            all_label.append(i)
            all_context.append(0)
    return all_data,all_label,all_context



class init_training_dataloader():
    def __init__(self, config, mean, std, balance_factor=1):
        super(init_training_dataloader, self).__init__()

        if config['dataset'] == "Cifar":
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            testing_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True) #transform=self.transform
            cifar_data = load_cifar10(testing_dataset)
            self.image, self.label, self.context = cifar_train(cifar_data)
            #print(testing_dataset)
            #exit()

        elif config['dataset'] == "NICO":
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.image, self.label, self.context = load_NICO(os.path.join(config['image_folder'], 'train'), balance_factor, config)

        elif config['dataset'] == "Imagenet9":
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.image, self.label, self.context = load_Imagenet9(os.path.join(config['image_folder'], 'train'), balance_factor, config)



    def get_dataloader(self, batch_size=16, num_workers=1, shuffle=True):
        training_dataset = NICO_dataset(self.image, self.label, self.context, transform=self.transform)
        training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return training_loader

    def get_pre_dataloader(self, batch_size=128, num_workers=1, shuffle=True, n_env=1):
        soft_split_init = torch.randn((len(self.image), n_env), device="cuda")
        # soft_split_init = torch.zeros((len(self.image), n_env), device="cuda").fill_(1/n_env)
        soft_split_init = torch.nn.Parameter(soft_split_init)
        optimizer = torch.optim.SGD([soft_split_init], lr=0.1, momentum=0.9, weight_decay=0)
        # optimizer = torch.optim.Adam([soft_split_init], lr=0.01, weight_decay=0)
        optimizer.zero_grad()
        optimizer.step()
        pre_scheduler = MultiStepLR(optimizer, [30], gamma=0.1, last_epoch=-1)
        training_dataset = NICO_dataset(self.image, self.label, self.context, transform=self.transform, soft_split=soft_split_init)
        training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return training_loader, optimizer, pre_scheduler

    def get_bias_dataloader(self, batch_size=128, num_workers=1, shuffle=True):
        training_dataset = NICO_dataset(self.image, self.label, self.context, transform=self.transform)
        training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return training_loader

    def get_env_dataloader(self, config, batch_size=16, num_workers=1, shuffle=True, pre_split=None):
        env = True
        n_env = config['variance_opt']['n_env']
        image_, label_, context_ = make_env(self.image, self.label, self.context, n_env, config['variance_opt']['env_type'], pre_split=pre_split)
        training_dataset = []
        training_dataset_all = NICO_dataset(self.image, self.label, self.context, transform=self.transform)

        for env_idx in range(n_env):
            training_dataset.append(NICO_dataset_env(image_[env_idx], label_[env_idx], context_[env_idx], env_idx, transform=self.transform, label2train=training_dataset_all.label2train))
        training_loader = []
        training_loader.append(DataLoader(CycleConcatDataset(*training_dataset), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size))

        if config['variance_opt']['mode'] == 'ours':
            if config['variance_opt']['erm_flag']:
                training_dataset_erm = training_dataset_all
                training_loader.append(DataLoader(training_dataset_erm, shuffle=shuffle, num_workers=num_workers, batch_size=n_env * batch_size))
        return training_loader



def get_test_dataloader(config, mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    if config['dataset'] == "Cifar":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        #cifar100_test = CIFAR100Test(path, transform=transform_test)
        testing_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

        cifar_data = load_cifar10(testing_dataset)
        image, label, context = cifar_test(cifar_data)
        val_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    elif config['dataset'] == "NICO":
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image, label, context = load_NICO(os.path.join(config['image_folder'], 'test'), config=config)
        testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)
    elif config['dataset'] == "Imagenet9":
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image, label, context = load_Imagenet9(os.path.join(config['image_folder'], 'test'), config=config)
        testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    testing_loader = DataLoader(
            testing_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return testing_loader


def get_val_dataloader(config, mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    if config['dataset'] == "Cifar":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        #cifar100_test = CIFAR100Test(path, transform=transform_test)
        #val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True) #transform=self.transform
        cifar_data = load_cifar10(val_dataset)
        image, label, context = cifar_val(cifar_data)
        val_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    elif config['dataset'] == "NICO":
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image, label, context = load_NICO(os.path.join(config['image_folder'], 'val'), config=config)
        val_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)
    elif config['dataset'] == "Imagenet9":
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image, label, context = load_Imagenet9(os.path.join(config['image_folder'], 'val'), config=config)
        val_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    val_loader = DataLoader(val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return val_loader


def cal_acc(outputs, labels):
    correct = 0.
    _, preds = outputs.max(1)
    #print(preds)
    #print(labels)
    #print(preds.eq(labels))
    correct += preds.eq(labels).sum()
    #print(correct)
    return correct, correct / labels.size(0)


def penalty(logits, y, loss_function):

    # scale = torch.tensor(1.).cuda().requires_grad_()
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

class Acc_Per_Context():
    def __init__(self, cxt_dic_path):
        super(Acc_Per_Context, self).__init__()
        import json
        self.cxt_dic = json.load(open(cxt_dic_path, 'r'))
        self.correct = {idx: 0 for idx in self.cxt_dic.values()}
        self.cnt = {idx:0 for idx in self.cxt_dic.values()}

    def update(self, outputs, labels, contexts):
        batch_size = outputs.size(0)
        for item in range(batch_size):
            output, label, context = outputs[item], labels[item], contexts[item]
            self.cnt[int(context)] += 1
            if output == label:
                self.correct[int(context)] += 1

    def cal_acc(self):
        acc_all = []
        for cxt in self.cxt_dic.keys():
            if self.cnt[self.cxt_dic[cxt]] > 0:
                acc_per_cxt = self.correct[self.cxt_dic[cxt]] / self.cnt[self.cxt_dic[cxt]]
                acc_all.append([cxt, acc_per_cxt])
        return acc_all


class Acc_Per_Context_Class():
    def __init__(self, cxt_dic_path, label_list):
        super(Acc_Per_Context_Class, self).__init__()
        import json
        self.cxt_dic = json.load(open(cxt_dic_path, 'r'))
        self.correct_cnt = {}
        for label in label_list:
            correct = {idx: 0 for idx in self.cxt_dic.values()}
            cnt = {idx:0 for idx in self.cxt_dic.values()}
            self.correct_cnt[label] = {'correct':correct, 'cnt':cnt}

    def update(self, outputs, labels, contexts):
        batch_size = outputs.size(0)
        for item in range(batch_size):
            output, label, context = outputs[item], labels[item], contexts[item]
            self.correct_cnt[int(label)]['cnt'][int(context)] += 1
            if output == label:
                self.correct_cnt[int(label)]['correct'][int(context)] += 1

    def cal_acc(self):
        acc_all = {}
        for label in self.correct_cnt.keys():
            acc_class = []
            for cxt in self.cxt_dic.keys():
                if  self.correct_cnt[label]['cnt'][self.cxt_dic[cxt]] > 0:
                    acc_per_cxt = self.correct_cnt[label]['correct'][self.cxt_dic[cxt]] / self.correct_cnt[label]['cnt'][self.cxt_dic[cxt]]
                    acc_class.append([cxt, acc_per_cxt])
            acc_all[label] = acc_class
        return acc_all


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def save_model(net, path):
    if isinstance(net, list):
        states = {}
        for idx, model in enumerate(net):
            states[idx] = model.state_dict()
        torch.save(states, path)
    else:
        torch.save(net.state_dict(), path)

def load_model(net, path):
    if isinstance(net, list):
        loaded_model = torch.load(path)
        for idx, model in enumerate(net):
            model.load_state_dict(loaded_model[idx])
    else:
        net.load_state_dict(torch.load(path))

def load_model_single(net, path):
    if isinstance(net, list):
        loaded_model = torch.load(path)
        for idx, model in enumerate(net):
            model.load_state_dict({k.replace('module.', ''): v for k, v in loaded_model[idx].items()})
    else:
        net.load_state_dict(torch.load(path))


def get_parameter_number(net):
    # debug: false
    if isinstance(net, list):
        total_num = trainable_num = 0
        for net_ in net:
            total_num += sum(p.numel() for p in net_.parameters())
            trainable_num += sum(p.numel() for p in net_.parameters() if p.requires_grad)
    else:
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('------------- Model Parameters ------------- ')
    print('Total: %.3f M \t Trainable: %.3f M' %(total_num/1e6, trainable_num/1e6))


class CycleConcatDataset(Dataset):
    '''Dataset wrapping multiple train datasets
    Parameters
    ----------
    *datasets : sequence of torch.utils.data.Dataset
        Datasets to be concatenated and cycled
    '''
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return max(len(d) for d in self.datasets)

#if __name__ == "__main__":
#    draw_pr()