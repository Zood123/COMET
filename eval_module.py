import torch
import torch.nn as nn
import os
import json
import tabulate
import random
import time
import torchvision
import torchvision.transforms as transforms
from utils import Acc_Per_Context, Acc_Per_Context_Class,show_image_one, load_Imagenet9, cal_acc, save_model, load_model,show_image,NICO_dataset,binarize_map,load_NICO,get_test_dataloader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from CAM import CAM_mapgeneration,ScoreCAM_generation
import attribution_methods
import torch.nn.functional as F
#import tensorflow as tf
from PIL import Image



def read_bam_mask(mask_fpath):
    #masks_mat = np.load(tf.gfile.GFile(mask_fpath, 'rb'), allow_pickle=True)
    masks_mat=np.load(tf.io.gfile.GFile(mask_fpath, 'rb'), allow_pickle=True)
    return masks_mat


ROW_MIN = 0
COL_MIN = 1
ROW_MAX = 2
COL_MAX = 3

# Same as in https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
RESNET_SHAPE = (224, 224)
IMG_SHAPE = (128, 128)
#from PIL import Image

def single_attr(map_2d, loc, obj_mask):
  """Given a 2D saliency map, the location of an object, and the binary mask of that object, compute the attribution of the object by averaging over its pixel-wise attributions."""

  obj_mask_resized = np.array(
      Image.fromarray(obj_mask).resize(
          (loc[COL_MAX] - loc[COL_MIN], loc[ROW_MAX] - loc[ROW_MIN]),
          Image.BILINEAR))
  # Create a new array with zeros
  final_mask = np.zeros((224, 224))  # Assuming obj_mask has the full image dimensions
  
  # Place the resized mask into the corresponding position in the final mask
  final_mask[loc[ROW_MIN]:loc[ROW_MAX], loc[COL_MIN]:loc[COL_MAX]] = obj_mask_resized
  
  avg = np.sum(
      map_2d[:, loc[ROW_MIN]:loc[ROW_MAX], loc[COL_MIN]:loc[COL_MAX]] *
      obj_mask_resized,
      axis=(-1, -2)) / np.count_nonzero(obj_mask_resized)
  return avg,final_mask


def bam_real_mask(sal_maps,locs):
  base_dir = "/home/xzz5508/code/Imbalance_ood/bam"
  data = "scene"
  mask_fpath = os.path.join(base_dir, 'data', data, 'val_mask')
  
  loc_fpath = os.path.join(base_dir, 'data', data, 'val_loc.txt')
  lines = [tf.io.gfile.GFile(loc_fpath, 'r').readlines()[i] for i in range(10000)]
  locs = np.array([[
      int(int(l) * float(RESNET_SHAPE[0]) / IMG_SHAPE[0])
      for l in line.rstrip('\n').split(' ')[-1].split(',')
  ]
                   for line in lines])
  '''
  '''
  #with open(os.path.join(base_dir, 'locs.pkl'), 'wb') as f:
  #  pickle.dump(locs, f)
  #exit()
  #pool = multiprocessing.Pool(num_threads)
  #maps_3d = np.array(sal_maps).reshape(-1, RESNET_SHAPE[0], RESNET_SHAPE[1], 3)
  #maps_2d = np.array(pool.map(visualize_pos_attr, maps_3d))
  maps_2d = sal_maps.reshape(
      len(sal_maps), int(1), RESNET_SHAPE[0],
      RESNET_SHAPE[1])
  #print(sal_maps.shape)
  #print(maps_2d.shape)
  #exit()

  file = open("/home/xzz5508/code/Imbalance_ood/bam/data/bam_scene_"+ "test" +".pkl", 'rb')
  data_addresses= pickle.load(file)
  file.close()
  img_names = []
  for class_label in data_addresses:
      for image_path in class_label:
          img_names.append(os.path.basename(image_path))


  #img_names = 
  if data in ['obj', 'scene', 'scene_only']:
    # MCS and IDR are evaluated on 10000 images and masks are 10x100.
    # Find the right mask.
    obj_dict = {'backpack': 0,
                'bird': 1,
                'dog': 2,
                'elephant': 3,
                'kite': 4,
                'pizza': 5,
                'stop_sign': 6,
                'toilet': 7,
                'truck': 8,
                'zebra': 9,
    }
    
    # Loading val_mask from the data directory
    masks_mat = np.load(tf.io.gfile.GFile(mask_fpath, 'rb'), allow_pickle=True)
    # Getting obj indices
    obj_inds = [obj_dict[i.split('.')[0].split('-')[0]] for i in img_names]
    # getting indices for a particular object class
    temp_inds = [int(i.split('.')[0][-2:]) for i in img_names]

    obj_masks = [masks_mat[obj_inds[i]*100 + temp_inds[i]]
                 for i, _ in enumerate(img_names)]
    
  
  attrs = []
  real_masks = []
  for i in range(len(maps_2d)):
    attr, real_mask = single_attr(maps_2d[i], locs[i], obj_masks[i])
    attrs.append(attr)
    real_masks.append(real_mask)
  
  #with open(os.path.join(base_dir, 'real_masks.pkl'), 'wb') as f:
  #  pickle.dump(real_masks, f)
  # COMET_bam_object_97
  with open(os.path.join(base_dir, 'COMET_bam_scene_scores.pkl'), 'wb') as f:
    pickle.dump(attrs, f)
  #print(attrs)
  exit()
  return obj_masks




def read_pickle(file_path):
    # Open the file in read-binary mode
    with open(file_path, 'rb') as file:
        # Load and return the data from the file
        data = pickle.load(file)
    return data



def calculate_auc(iou_scores, step=0.05):
    thresholds = np.arange(0, 1 + step, step)
    cumulative_scores = [np.mean([iou >= th for iou in iou_scores]) for th in thresholds]
    #print(cumulative_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, cumulative_scores, marker='o', linestyle='-', color='b')
    plt.title('Cumulative IoU Scores vs. IoU Thresholds')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Proportion of Samples Exceeding Threshold')
    plt.grid(True)
    plt.savefig("test_iou_curve.png")
    plt.close()

    
    #exit()
    auc = np.trapz(cumulative_scores, thresholds)  # Use trapezoidal rule to approximate the area under curve
    return auc



@torch.no_grad()
def eval_training(config, args, net, test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    acc_per_context = Acc_Per_Context(config['cxt_dic_path'])

    for i, (images, labels, context) in enumerate(test_loader):

        images = images.cuda()
        labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if "COMET" in config['exp_name']:
                outputs,mask,comp = net.forward_all(images)
            else:
                outputs = net(images)

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        if "COMET" in config['exp_name'] and i==0:
            dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/" +config['exp_name']+"/set"+str(i)+"/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            for j,image_show in enumerate(images):
                one_set = dir+"_"+str(j)+"_"
                # "number"+ str(j)+
                image_show  = image_show.permute(1, 2, 0).cpu().numpy()
                one_mask = mask[j].permute(1, 2, 0).cpu().numpy()
                #mixed_image = mixed_images[j].permute(1, 2, 0).cpu().numpy()
                #scaled_mask = upscale_mask(one_mask)
                #scaled_mask = np.stack((scaled_mask,) * 3, axis=-1)
                masked_image =  image_show*one_mask
                one_mask = np.squeeze(one_mask)
                show_image(masked_image,one_mask,image_show,one_set,std= config['training_opt']['std'],mean = config['training_opt']['mean'])

        if config["dataset"] != "Imagenet9":
            acc_per_context.update(preds, labels, context)

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    if config["dataset"] != "Imagenet9":
        print('Evaluate Acc Per Context...')
        acc_cxt = acc_per_context.cal_acc()
        print(tabulate.tabulate(acc_cxt, headers=['Context', 'Acc'], tablefmt='grid'))

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)





#@torch.no_grad()
def B_cos_attribution(config, args, net ,load_path):
    start = time.time()

    load_model(net, load_path)
    # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    
    #
    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    #gt_masks = read_pickle("/home/xzz5508/code/Imbalance_ood/an8Flower/Flowerdouble_masks_test.pkl")
    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)


    image, label, context = load_Imagenet9("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)
    image_b, label_b, context_b = load_Imagenet9("test",Data_version="im_mixed_next")
    testing_dataset_b = NICO_dataset(image_b, label_b, context_b, transform_test, require_context=True)

    '''
    bb_model = torchvision.models.resnet101(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet101/resnet101-194-best.pth"
    
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth" #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    acc_all = 0
    '''


    only_positive = True
    binarize = False
    interpolate = False
    attributor = attribution_methods.get_attributor(
            net, "BCos", only_positive, binarize, interpolate, (224, 224), batch_mode=False)


    count = 0 
    maskin_saveroot = "/home/xzz5508/code/images_gallory_bam/Bcos_object"

    total_time = 0


    masks = []
    IOU_list = []
    """
    for i,onecls_mask in enumerate(gt_masks):
        print(i)
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        for j,gt_mask in enumerate(onecls_mask):
    """
    for i in range(10):
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        for j in range(1000):
            img, label,context =testing_dataset[count]
            img_b, label_b,context_b  = testing_dataset_b[count]
            img_b = img_b.cuda().requires_grad_().unsqueeze(0)

            count = count+1

            #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
            if args.gpu:
                img = img.cuda().requires_grad_().unsqueeze(0)
                #label = label.cuda()
            
            #img = add_random_mask_pytorch(img,0.8)
            time_start = time.time()
            outputs = net(img)
            _, preds = outputs.max(1)
            correct += preds.eq(label).sum()
            mask = attributor(img, outputs, preds, 0)


            reference_mask = mask > torch.mean(mask)

            time_end = time.time()
            time_used = time_end - time_start
            total_time = total_time +time_used
            #mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
            # 224*224
            #img_m = add_random_mask_pytorch(img,0.8)
            outputs = net(img_b)
            _, preds = outputs.max(1)
            correct += preds.eq(label).sum()
            mask = attributor(img_b, outputs, preds, 0)
            masks.append(mask)
            #mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
            #show_image_one(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
            #print(mask.shape)

            #mask = F.interpolate(mask, size=(len(gt_mask), len(gt_mask)), mode='bilinear', align_corners=False)
            mask =mask.squeeze(dim=0).squeeze(dim=0)
            mask = mask.cpu().detach().numpy()
            #print(mask.shape)

            '''
            mask = binarize_map(mask,0.5,complement=False) # 0.2005 maskin 
            maskedin_img = mask*img
            acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
            acc_all = acc_all +acc
            '''
            
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated.png', mask.cpu())
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.png', gt_mask)
            #plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
            #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)


            '''
            binarized_mask = mask > 0.1
            intersection = np.logical_and(binarized_mask, gt_mask)
            union = np.logical_or(binarized_mask, gt_mask)
            iou = np.sum(intersection) / np.sum(union)
            IOU_list.append(iou)            
            '''
            
            #evaluator.accumulate(mask,reference_mask.squeeze(dim=0).squeeze(dim=0).cpu().detach().numpy())
            #if j ==99:
            #    print(total_time/100)
            #    exit()

    all_masks = torch.stack(masks)
    torch.save(all_masks, 'Bcos_bamobject.pt')

    #acc_all = acc_all/count
    #print(acc_all)    

    #auc= evaluator.compute()
    #print(auc)
    #auc = calculate_auc(IOU_list,1/20000)
    #print(auc)
    exit()



#@torch.no_grad()
def B_cos_attribution_nico(config, args, net ,load_path):
    start = time.time()

    load_model(net, load_path)
    # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    #
    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)


    image, label, context = load_NICO("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)


    acc_all = 0
    only_positive = True
    binarize = False
    interpolate = False
    attributor = attribution_methods.get_attributor(
            net, "BCos", only_positive, binarize, interpolate, (224, 224), batch_mode=False)


    count = 0 
    maskin_saveroot = "/home/xzz5508/code/baseline_gallory/BCos"

    total_time = 0
    for (img, label, context) in testing_dataset:
        
        count = count+1
        if args.gpu:
            img = img.cuda().requires_grad_().unsqueeze(0)
            #label = label.cuda()
            
        #img = add_random_mask_pytorch(img)
        time_start = time.time()
        outputs = net(img)

        _, preds = outputs.max(1)
        #correct += preds.eq(label).sum()
            
        mask = attributor(img, outputs, torch.tensor([label]).cuda(), 0).detach().squeeze(0).squeeze(0)
        time_end = time.time()
        time_used = time_end - time_start
        total_time = total_time +time_used
        #mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
        # 224*224

        #show_image_one(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
        #print(mask.shape)
        #mask =mask.squeeze(dim=0).squeeze(dim=0)
        #mask = mask.cpu().numpy()
        #print(mask.shape)

            
        mask = binarize_map(mask,0.2,complement=True) # 0.2005 maskin 
        maskedin_img = mask*img
        acc = eval_postacc(maskedin_img,net,torch.tensor([label]).cuda())
        acc_all = acc_all +acc
        '''
        '''

    acc_all = acc_all/count
    print(acc_all)    

    #auc= evaluator.compute()
    #print(auc)
    exit()





@torch.no_grad()
def RBF_eval(config, args, net ,load_path):
    start = time.time()
    load_model(net.selector,load_path)



    #load_model(net, load_path)
    # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    #
    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    #gt_masks = read_pickle("/home/xzz5508/code/Imbalance_ood/an8Flower/Flowerdouble_masks_test.pkl")

    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)


    image, label, context = load_Imagenet9("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)


    image_b, label_b, context_b = load_Imagenet9("test",Data_version="im_mixed_next")

    testing_dataset_b = NICO_dataset(image_b, label_b, context_b, transform_test, require_context=True)
    '''
    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = ""
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_xreal/RBF_trainpredictor-27-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet101/resnet101-194-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_xreal/RBF_trainpredictor-27-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_fixed/RBF_trainpredictor-58-best.pth" #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    '''
    acc_all = 0



    
    maskin_saveroot = "/home/xzz5508/code/images_gallory_flower/double/RBF"
    #exit()

    IOU_list = []
    count = 0 
    time_total = 0
    for i,onecls_mask in enumerate(gt_masks):
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        print(i)
        for j,gt_mask in enumerate(onecls_mask):
            img, label,context =testing_dataset[count]
            img_b, label_b,context_b  = testing_dataset_b[count]
            count = count+1
            

            img_b = img_b.cuda().requires_grad_().unsqueeze(0)
            #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
            if args.gpu:
                img = img.cuda().requires_grad_().unsqueeze(0)
                #label = label.cuda()
            
            
            #img = add_random_mask_pytorch(img,0.8)
            
            
            #outputs = net(img)

            #_, preds = outputs.max(1)
            #correct += preds.eq(label).sum()
            
            time_start = time.time()
            mask = net.training_step(img, label, 0,eval=True) #torch.Size([1, 1, 224, 224])
            time_end = time.time()
            time_used = time_end-time_start
            time_total=time_total+time_used



            reference_mask = mask > torch.mean(mask)
            mask = net.training_step(add_random_mask_pytorch(img,0.8), label, 0,eval=True) 
            #print(mask.shape)
            #print(mask)
            #print(torch.sum(mask))
            
            #show_image_one(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
            #print(mask.shape)
            #mask = F.interpolate(mask, size=(len(gt_mask), len(gt_mask)), mode='bilinear', align_corners=False)
            mask =mask.squeeze(dim=0).squeeze(dim=0)
            mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
            mask = mask.detach().cpu().numpy()
            #print(mask.shape)

            
            #mask = binarize_map(mask,0.05,complement=True)
            #maskedin_img = mask*img
            #acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
            #acc_all = acc_all +acc
            '''
            '''
            #plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
            #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated.png', mask.detach().cpu())
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.png', gt_mask)


            '''
            binarized_mask = mask > 0.5
            intersection = np.logical_and(binarized_mask, gt_mask)
            union = np.logical_or(binarized_mask, gt_mask)
            iou = np.sum(intersection) / np.sum(union)
            IOU_list.append(iou)           
            '''
            evaluator.accumulate(mask,reference_mask.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            #if j ==99:
            #    print(time_total/100)
            #    exit()

    #acc_all = acc_all/count
    #print(acc_all)    

    auc= evaluator.compute()
    print(auc)
    #auc = calculate_auc(IOU_list,1/20000)
    #print(auc)

    exit()



@torch.no_grad()
def RBF_eval_nico(config, args, net ,load_path):
    start = time.time()


    load_model(net.selector,load_path)
    #load_model(net, load_path)
    # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    #
    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)


    image, label, context = load_NICO("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_realx_nico/RBF_trainpredictor-35-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_fixed_nico/RBF_trainpredictor-35-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_fixed_nico/RBF_trainpredictor-35-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_realx_nico/RBF_trainpredictor-35-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"
    #
    # "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_realx_nico/RBF_trainpredictor-35-best.pth"
    
    
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainpredictor_resnet18_fixed/RBF_trainpredictor-58-best.pth" #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    acc_all = 0



    
    maskin_saveroot = "/home/xzz5508/code/baseline_gallory/rbf"
    #exit()

    correct = 0
    count = 0 
    time_total = 0
    for (img, label, context) in testing_dataset:
        img, label,context =testing_dataset[count]
        
        count = count+1
        #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
        if args.gpu:
            img = img.cuda().requires_grad_().unsqueeze(0)
            #label = label.cuda()
            
            
        #img = add_random_mask_pytorch(img)
            
            
        #outputs = net(img)

        #_, preds = outputs.max(1)
        #correct += preds.eq(label).sum()
         
        time_start = time.time()
        mask = net.training_step(img, label, 0,eval=True) #torch.Size([1, 1, 224, 224])
        #mask = generate_random_mask([224,224]).cuda()
        time_end = time.time()
        time_used = time_end-time_start
        time_total=time_total+time_used

        #_, preds = outputs.max(1)
        #correct += preds.eq(label).sum()

        #print(mask.shape)
        #print(mask)
        #print(torch.sum(mask))
            
        #show_image_one(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
        #print(mask.shape)
        #mask =mask.squeeze(dim=0).squeeze(dim=0)
        mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
        #mask = mask.detach().cpu().numpy()
        #print(mask.shape)

            
        mask = binarize_map(mask,0.1,complement=True)
        maskedin_img = mask*img
        #maskedin_img = img
        acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
        acc_all = acc_all +acc
        '''
        '''
        #plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
        #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated.png', mask.detach().cpu())
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.png', gt_mask)
        #evaluator.accumulate(mask,gt_mask)
        #if j ==99:
        #    print(time_total/100)
        #    exit()    

    #acc_all = acc_all/count
    acc_all = acc_all/count
    print(acc_all)    

    #auc= evaluator.compute()
    #print(auc)
    exit()

@torch.no_grad()
def posthoc_eval(config, args, net, explandum,test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error

    for i, (images, labels, context) in enumerate(test_loader):

        images = images.cuda()
        labels = labels.cuda()
        mask = net(images,labels)
        loss =loss_function.get(mask,images,labels,explandum)
        test_loss += loss.item()
        dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/posthoc/" +config['exp_name']+"/set"+str(i)+"/"

        

        if i==0:
            if not os.path.exists(dir):
                os.makedirs(dir)
            for j,image_show in enumerate(images):
                one_set = dir+"_"+str(j)+"_"
                # "number"+ str(j)+
                image_show  = image_show.permute(1, 2, 0).cpu().numpy()
                one_mask = mask[j].permute(1, 2, 0).cpu().numpy()
                #mixed_image = mixed_images[j].permute(1, 2, 0).cpu().numpy()
                #scaled_mask = upscale_mask(one_mask)
                #scaled_mask = np.stack((scaled_mask,) * 3, axis=-1)
                masked_image =  image_show*one_mask
                one_mask = np.squeeze(one_mask)
                show_image(masked_image,one_mask,image_show,one_set,std= config['training_opt']['std'],mean = config['training_opt']['mean'])
    
    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        finish-start
    ))


    #add informations to tensorboard
    #if tb:
    #    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    #    #writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return test_loss / len(test_loader.dataset)



def add_random_mask_pytorch(batch_images, mask_probability=0.9):
    """
    Add a random binary mask to a batch of images (PyTorch version).

    Parameters:
    - batch_images: Input batch of images (PyTorch tensor with shape (batch_size, channels, height, width)).
    - mask_probability: Probability of including a pixel in the mask.

    Returns:
    - Batch of images with a random binary mask applied.
    """
    mask = (torch.rand_like(batch_images[:, 0:1, :, :]) < mask_probability).float()
    masked_images = batch_images * mask

    return masked_images






@torch.no_grad()
def eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch):
    start = time.time()
    try:
        #load_model(net, checkpoint_path.format(net=args.net, epoch=best_epoch, type='best'))
        load_model(net, checkpoint_path)
        
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=best_epoch, type='best')))
    except:
        print('no best checkpoint')
        load_model(net, checkpoint_path.format(net=args.net, epoch=180, type='regular'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for i,(images, labels, context) in enumerate(test_loader):

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if "COMET" in config['exp_name']:
                outputs,mask,comp = net.forward_all(images)
            else:
                outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        if "COMET" in config['exp_name'] and i==0:
            dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/" +config['exp_name']+"/test_set"+str(i)+"/"
            #dir =  "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/nico_gallory/3p20entropy/test_set"+str(i)+"/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            for j,image_show in enumerate(images):
                one_set = dir+"_"+str(j)+"_"
                # "number"+ str(j)+
                image_show  = image_show.permute(1, 2, 0).cpu().numpy()
                one_mask = mask[j].permute(1, 2, 0).cpu().numpy()
                #mixed_image = mixed_images[j].permute(1, 2, 0).cpu().numpy()
                #scaled_mask = upscale_mask(one_mask)
                #scaled_mask = np.stack((scaled_mask,) * 3, axis=-1)
                masked_image =  image_show*one_mask
                one_mask = np.squeeze(one_mask)
                show_image(masked_image,one_mask,image_show,one_set,std= config['training_opt']['std'],mean = config['training_opt']['mean'])

        if config["dataset"] != "Imagenet9":
            acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    if config["dataset"] != "Imagenet9":
        print('Evaluate Acc Per Context Per Class...')
        class_dic = json.load(open(config['class_dic_path'], 'r'))
        class_dic = {v: k for k, v in class_dic.items()}
        acc_cxt_all_class = acc_per_context.cal_acc()
        for label_class in acc_cxt_all_class.keys():
            acc_class = acc_cxt_all_class[label_class]
            print('Class: %s' %(class_dic[int(label2train[label_class])]))
            print(tabulate.tabulate(acc_class, headers=['Context', 'Acc'], tablefmt='grid'))

    return correct.float() / len(test_loader.dataset)


@torch.no_grad()
def eval_mode(config, args, net, test_loader, loss_function, model_path,explaindum=None):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    #label2train = test_loader.dataset.label2train
    #label2train = {v: k for k, v in label2train.items()}
    #acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for (images, labels, context) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if "COMET" in config['exp_name']:
                outputs,mask,comp = net.forward_all(images)
            else:
                if explaindum != None:
                    out = explaindum(images)
                    _,pred = out.max(1)
                    mask = net(images,pred)
                    outputs = explaindum(mask*images)
                else:
                    outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        #if config["dataset"] != "Imagenet9":
        #    acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print(correct.float())
    return correct.float() / len(test_loader.dataset)



def read_mask(mask_path,datapath_pickle):
    file = open(datapath_pickle, 'rb')
    data_addresses= pickle.load(file)
    file.close()

    gt_mask = [ [] for x in data_addresses]
    for i,oneclass in enumerate(data_addresses):
        class_name = oneclass[0].split("/")[-2]
        for one_sample in oneclass:
            image_name = one_sample.split("/")[-1].split('.')[0]
            path = mask_path+"/"+class_name+"/"+image_name+".npy"
            image = np.load(path)
            gt_mask[i].append(image)
    return gt_mask


from evaluation_explain import MaskEvaluator

def calculate_average_attribution(attribution_map, ground_truth_mask):
    """
    Calculate the average attribution of pixels within the ground truth mask region.
    
    Parameters:
    - attribution_map (numpy.ndarray): The attribution map where each pixel's value represents its importance.
    - ground_truth_mask (numpy.ndarray): A binary mask where pixels of interest are marked as 1 (or True).

    Returns:
    - float: The average attribution of the pixels specified by the ground truth mask.
    """
    # Ensure the ground truth mask is boolean for proper masking
    ground_truth_mask = ground_truth_mask.astype(bool)
    
    # Apply the ground truth mask to filter the attribution map
    masked_attributions = attribution_map[ground_truth_mask]
    
    # Calculate the average attribution over the masked region
    average_attribution = np.mean(masked_attributions)
    
    return average_attribution
    

@torch.no_grad()
def eval_explain(config, args, net, test_loader, loss_function, model_path):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()
    # original 
    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    #gt_masks = read_pickle("/home/xzz5508/code/Imbalance_ood/an8Flower/Flowerdouble_masks_test.pkl")


    confidence_list = [i / 20.0 for i in range(1, 20)]     
    evaluator =MaskEvaluator(confidence_list)

    image, label, context = load_Imagenet9("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    #image_b, label_b, context_b = load_Imagenet9("test",Data_version="im_mixed_next")
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    



    '''
    # Extract scores using the correct indices
    correct_scores = [scene_scores[idx] for idx in correct_index_scene if idx < len(scene_scores)]

    # Calculate the average of these scores
    if correct_scores:  # Check if the list is not empty
        average_score = sum(correct_scores) / len(correct_scores)
    else:
        average_score = None  # or appropriate fallback value, such as 0 or an error message

    print("Average Score:", average_score)
    #exit()
    
    '''
    # Ensure both lists have the same length and corresponding elements
    '''
    assert len(scene_scores) == len(COMET_bam_scene_only_scores), "The score lists should be of the same length."

    # Count how many times the scores in COMET_bam_scene_only_scores are greater than those in scene_scores
    count_greater = sum(1 for i in range(len(scene_scores)) if COMET_bam_scene_only_scores[i] > scene_scores[i])

    # Calculate the rate
    rate_greater = count_greater / len(scene_scores)

    print("Rate at which COMET_bam_scene_only_scores are greater:", rate_greater)
    '''

    '''
    # Calculate the differences between corresponding scores
    #score_differences = [obj - scene for obj, scene in zip(obj_scores, scene_scores)]

    # Calculate the average of the score differences
    #average_difference = sum(score_differences) / len(score_differences)

    #print("Average score difference:", average_difference)
    #exit()
    '''

    #/home/xzz5508/code/Imbalance_ood/bam/data/COMET_bam_object_97_scores.pkl

    #gt_masks_1 = read_bam_mask("/home/xzz5508/code/Imbalance_ood/bam/data/obj/val_mask")
    
    
    #exit()
    #"/home/xzz5508/code/Imbalance_ood/bam/data/obj/val_mask")
    #print(len(gt_masks))
    #print(gt_masks[0])
    #print(len(gt_masks[0]))
    #print(len(gt_masks[0][0][0]))
    #exit()
    count = 0
    for i,onecls_mask in enumerate(gt_masks):
        for j,gt_mask in enumerate(onecls_mask):
            
            img, label,context =testing_dataset[count]
            #img_b, label_b,context_b  = testing_dataset_b[count]
            #img_b = img_b.cuda().unsqueeze(0)
            #print(img.permute(1, 2, 0).numpy().shape)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png', img.permute(1, 2, 0).numpy())
            #show_image_one(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
            #gt_mask = real_masks[count]

            img = img.cuda().unsqueeze(dim=0)
            #gt_mask= gt_masks_1[j]
            #img = add_random_mask_pytorch(img,0.8)
            #print(len(loaded_masks_o))
            #exit()
            '''
            mask_o = loaded_masks_o[count]
            mask_s = loaded_masks_s[count]
            mask_truth = gt_masks[count]

            mask_o = F.interpolate(mask_o, size=(len(mask_truth), len(mask_truth[0])), mode='bilinear', align_corners=False)
            mask_s = F.interpolate(mask_s, size=(len(mask_truth), len(mask_truth[0])), mode='bilinear', align_corners=False)

            mask_o =mask_o.squeeze(dim=0).squeeze(dim=0)
            mask_o = mask_o.cpu().numpy()
            mask_s =mask_s.squeeze(dim=0).squeeze(dim=0)
            mask_s = mask_s.cpu().numpy()
            o_score = calculate_average_attribution(mask_o,mask_truth)
            s_score = calculate_average_attribution(mask_s,mask_truth)
            mcs_all = mcs_all + o_score-s_score
            '''

            #exit()
            
            
            outputs,mask,comp =net.forward_all(img)
            
            
            #reference_mask = mask > torch.mean(mask)
            #outputs,mask,comp  = net.forward_all(img_b)
            #masks.append(mask)

            #outputs = bb_model(img)
            #mask = torch.ones_like(mask)
            #mask =(mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
            #time_total = time_total+ time_used
            #mask = generate_gaussian_score_map([224,224])
            #mask = generate_random_mask([224,224]).cuda()
            #print(mask.shape)
            
            #acc, _ = cal_acc(outputs, torch.tensor([label]).cuda())
            #if acc==1.00:
            #    correct_index.append(count)

            #print(acc)
            #exit()
            
            #mask = binarize_map(mask,0.05,complement=True) #0.2,complement=True)
            #maskedin_img = mask*img
            #acc = eval_postacc(maskedin_img, net.predictor ,torch.tensor([label]).cuda(),COMET=False) #bb_model , net
            # torch.tensor([label]).cuda()
            #acc_all = acc_all +acc
            '''
            '''

            # for flower
            #mask = F.interpolate(mask, size=(len(gt_mask), len(gt_mask)), mode='bilinear', align_corners=False)
            mask =mask.squeeze(dim=0).squeeze(dim=0)
            mask = mask.cpu().numpy()

            
            '''
            binarized_mask = mask > 0.9
            intersection = np.logical_and(binarized_mask, gt_mask)
            union = np.logical_or(binarized_mask, gt_mask)
            iou = np.sum(intersection) / np.sum(union)
            IOU_list.append(iou)
            '''
            
            #print(iou)
            #exit()
            evaluator.accumulate(mask,gt_mask) #reference_mask.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            count = count +1
            #if j ==99:
            #   print(time_total/100)
            #   exit()
    

    #all_masks = torch.stack(masks)
    #torch.save(all_masks, 'COMET_bam_scene_89.pt')
    
    #with open('correct_index_scene.pkl', 'wb') as f:
    #    pickle.dump(correct_index, f)
    
    #correct_index
    #mcs_all=mcs_all/count
    #print(mcs_all)
    #acc_all = acc_all/count
    #print(acc_all)
    auc= evaluator.compute()
    print(auc)

    #auc = calculate_auc(IOU_list,1/20000)
    #print(auc)

    exit()

@torch.no_grad()
def eval_explain_NICO(config, args, net, test_loader, loss_function, model_path):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()
    # original 
    '''
    image, labels, context = load_NICO(os.path.join(config['image_folder'], 'test'), config=config)
    
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    testing_dataset = NICO_dataset(image, labels, context, transform_test, require_context=True)
    '''
    test_loader = get_test_dataloader(
        config,
        config['training_opt']['mean'],
        config['training_opt']['std'],
        num_workers=4,
        batch_size= 64,  # config['training_opt']['batch_size']
        shuffle=True # default: false
    )

    bb_model = torchvision.models.resnet101(pretrained=False).cuda()
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num']).cuda()
    
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet101_nico_3/resnet101-96-best.pth"
    load_model(bb_model,bbweights_path)
    bb_model.eval()
    acc_all = 0

    maskin_saveroot = "/home/xzz5508/code/images_gallory_nico/"
    '''
    '''
    correct = 0
    for i, (img, label, context) in enumerate(test_loader):
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        if args.gpu:
            img = img.cuda()
            label = label.cuda()
        
        #img = add_random_mask_pytorch(img)
        #outputs = bb_model(img)
        _,mask,comp =net.forward_all(img)
        #mask =mask.squeeze(dim=0).squeeze(dim=0)
        #mask = binarize_map(mask,0.05,complement=True)
        maskedin_img = mask*img
        #maskedin_img = img
        #maskedout_img =  (1-mask)*img
        #outputs = bb_model(maskedin_img)
        for j in range(len(img)):
            show_image_one(img[j].permute(1, 2, 0).cpu().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
            show_image_one(maskedin_img[j].squeeze().permute(1, 2, 0).cpu().numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"masked"+str(j)+".png")
            print(mask[j].shape)
            plt.imsave(cls_dir+"mask"+str(j)+".png", mask[j].squeeze(dim=0).cpu().numpy())



       #outputs = net.predictor(maskedin_img)
        #_, preds = outputs.max(1)
        #correct += preds.eq(label).sum()
        #mask = mask.cpu().numpy()
    #print(correct)
    acc_all = correct/len(test_loader.dataset)
    #print(len(test_loader.dataset))
    print(acc_all)
    #auc= evaluator.compute()
    #print(auc)
    exit()





@torch.no_grad()
def posthoc_explain(config, args, net, explandum, model_path):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    #gt_masks = read_pickle("/home/xzz5508/code/Imbalance_ood/an8Flower/Flowerdouble_masks_test.pkl")
    
    
    
    confidence_list = [i / 40.0 for i in range(1, 40)] # second 10 wrong !
    evaluator =MaskEvaluator(confidence_list)

    
    image, label, context = load_Imagenet9("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    '''
    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet101/resnet101-194-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth" # "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_imagenet_dropout/resnet18-71-best.pth"#"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    '''
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)
    image_b, label_b, context_b = load_Imagenet9("test",Data_version="im_mixed_next")

    testing_dataset_b = NICO_dataset(image_b, label_b, context_b, transform_test, require_context=True)
    
    IOU_list  = []
    print(len(image))
    #print(len(all_data[0]))
    #print((all_data[0]))
    #exit()
    count = 0 
    acc_all = 0
    time_total = 0
    maskin_saveroot = "/home/xzz5508/code/images_gallory_flower/double/posthoc"
    for i,onecls_mask in enumerate(gt_masks):
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        for j,gt_mask in enumerate(onecls_mask):
            
            img, label,context =testing_dataset[count]
            img_b, label_b,context_b =testing_dataset_b[count]
            #print(img.permute(1, 2, 0).numpy().shape)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png', img.permute(1, 2, 0).numpy())
            #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
            #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
            img = img.cuda().unsqueeze(dim=0)
            img_b = img_b.cuda().unsqueeze(dim=0)
            #img = add_random_mask_pytorch(img,0.8)
            count = count +1
            #outputs,mask,comp =net.forward_all(img.cuda().unsqueeze(dim=0))
            time_start = time.time()
            out = explandum(img)
            _,pred = out.max(1)
            mask = net(img,pred)
            reference_mask = mask > torch.mean(mask)
            time_end = time.time()
            time_used = time_end-time_start
            time_total = time_total+time_used

            #img_n = add_random_mask_pytorch(img,0.8)
            out = explandum(img_b)
            _,pred = out.max(1)
            mask = net(img_b,pred)
            #mask = generate_gaussian_score_map([224,224])
            #mask = generate_random_mask([224,224]).cuda()
            
            #print(mask.shape)
            #mask = F.interpolate(mask, size=(len(gt_mask), len(gt_mask)), mode='bilinear', align_corners=False)
            mask = mask.squeeze(dim=0).squeeze(dim=0)
            mask = mask.cpu().numpy()
            

            
             
            #mask = binarize_map(mask,0.05,complement=True)
            #maskedin_img = mask*img.cuda()
            #maskedout_img =  (1-mask)*img
            #acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
            #acc_all = acc_all +acc

            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated0.2.png', mask)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.png', gt_mask)
            #plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
            #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)

            '''
            binarized_mask = mask > 0.95
            intersection = np.logical_and(binarized_mask, gt_mask)
            union = np.logical_or(binarized_mask, gt_mask)
            iou = np.sum(intersection) / np.sum(union)
            IOU_list.append(iou)
            '''
            evaluator.accumulate(mask,reference_mask.squeeze(dim=0).squeeze(dim=0).cpu().numpy())
            #if j ==99:
            #    print(time_total/100)
            #    exit()

    auc= evaluator.compute()
    print(auc)
    #acc_all = acc_all/count
    #print(acc_all)
    #print("posthoc——")
    #auc = calculate_auc(IOU_list,1/20000)
    #print(auc)
    exit()



@torch.no_grad()
def posthoc_explain_NICO(config, args, net, explandum, model_path):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    confidence_list = [i / 40.0 for i in range(1, 40)] # second 10 wrong !
    evaluator =MaskEvaluator(confidence_list)

    
    image, label, context = load_NICO("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_nico_2/resnet18-173-best.pth"
    
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth" # "#"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()

    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    print(len(image))
    #print(len(all_data[0]))
    #print((all_data[0]))
    #exit()
    count = 0 
    acc_all = 0
    time_total = 0
    maskin_saveroot = "/home/xzz5508/code/baseline_gallory/posthoc"
    for (img, label, context) in testing_dataset:


        #print(img.permute(1, 2, 0).numpy().shape)
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png', img.permute(1, 2, 0).numpy())
        #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir ="/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png')
        #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
        img = img.cuda().unsqueeze(dim=0)
        #img = add_random_mask_pytorch(img)
        count = count +1
        #outputs,mask,comp =net.forward_all(img.cuda().unsqueeze(dim=0))
        #time_start = time.time()
        out = explandum(img)
        _,pred = out.max(1)
        mask = net(img,pred)
        #time_end = time.time()
        #time_used = time_end-time_start
        #time_total = time_total+time_used
        #mask = generate_gaussian_score_map([224,224])
        #mask = generate_random_mask([224,224])
        mask = mask.cuda()
        #print(mask.shape)
        
        #mask =mask.squeeze(dim=0).squeeze(dim=0)    
        #mask = mask.cpu().numpy()
        
        
         
        mask = binarize_map(mask,0.1,complement=True)
        maskedin_img = mask*img.cuda()

        #maskedout_img =  (1-mask)*img
        acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
        acc_all = acc_all +acc 
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated.png', mask)
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.png', gt_mask)
        #plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
        #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)

        #evaluator.accumulate(mask,gt_mask)
        #if j ==99:
        #    print(time_total/100)
        #    exit()


    acc_all = acc_all/count
    print(acc_all)
    exit()







def generate_gaussian_score_map(image_size, sigma=1):
    # Create a grid of coordinates centered at the image
    x = torch.linspace(-image_size[1] // 2, image_size[1] // 2, image_size[1])
    y = torch.linspace(-image_size[0] // 2, image_size[0] // 2, image_size[0])
    xv, yv = torch.meshgrid(x, y)

    # Calculate the Gaussian distribution
    gaussian_map = torch.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    gaussian_map = gaussian_map / (2 * np.pi * sigma**2)  # Normalize

    return gaussian_map


def generate_random_mask(mask_size):
    # Generate a random mask with values uniformly distributed in [0, 1]
    random_mask = torch.rand(mask_size)

    return random_mask




@torch.no_grad()
def eval_CAM(config, args, net):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()


    maskin_saveroot = "/home/xzz5508/code/images_gallory_flower/double/Score-CAM"


    #gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    gt_masks = read_pickle("/home/xzz5508/code/Imbalance_ood/an8Flower/Flowerdouble_masks_test.pkl")
    
    
    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)

    
    image, label, context = load_Imagenet9("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)
    
    '''
    image_b, label_b, context_b = load_Imagenet9("test",Data_version="im_mixed_next") 
    testing_dataset_b = NICO_dataset(image_b, label_b, context_b, transform_test, require_context=True)
    '''
    '''
    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_shuffled_im/resnet18-180-regular.pth"
    
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth" 
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_im9_2/resnet18-183-best.pth" 
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth" # "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_imagenet_dropout/resnet18-71-best.pth"#"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    '''
    #print(len(image))
    #print(len(all_data[0]))
    #print((all_data[0]))
    #exit()
    count = 0 
    acc_all = 0
    total_time = 0
    masks = []

    loaded_masks = torch.load('ScoreCam_masks_flower_double.pt')
    #loaded_masks_b = torch.load('ScoreCam_masks.pt')
    
    # /home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/ScoreCam_masks_im2.pt
    # /home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/Cam_masks_im.pt
    #/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/ScoreCam_masks.pt
    # /home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/ScoreCam_masks_im3.pt


    IOU_list = []
    for i,onecls_mask in enumerate(gt_masks):
        cls_dir =maskin_saveroot+"/"+str(i)+"/"
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        print(i)
        for j,gt_mask in enumerate(onecls_mask):
            img, label,context =testing_dataset[count]
            #img_b, label_b,context_b =testing_dataset_b[count]
            
            #show_image_one(img.permute(1, 2, 0).numpy(),std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"original"+str(j)+".png")
            img = img.cuda().unsqueeze(dim=0)
            #img_b = img_b.cuda().unsqueeze(dim=0)
            #print(img.permute(1, 2, 0).numpy().shape)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png', img.permute(1, 2, 0).numpy())
            #show_image()
            
            #img = add_random_mask_pytorch(img,0.8)
            mask = loaded_masks[count]
            #mask_b = loaded_masks_b[count]
            #reference_mask = mask_b > torch.mean(mask_b)
            count = count +1
            #time_start = time.time()
            #mask= ScoreCAM_generation(net,img) #,len(gt_mask)
            mask = F.interpolate(mask, size=(len(gt_mask), len(gt_mask)), mode='bilinear', align_corners=False)
            #mask = CAM_mapgeneration(net,img,len(gt_mask))
            #reference_mask = mask > torch.mean(mask)
            

            #mask = CAM_mapgeneration(net,img_b,len(gt_mask))
            #time_end = time.time()
            #time_used = time_end-time_start
            #total_time = total_time +time_used
            #print(mask.shape)
            #masks.append(mask)

            #print(mask.shape)
            #exit()
            
            mask =mask.squeeze(dim=0).squeeze(dim=0)
            mask = mask.cpu().numpy()




            #mask = binarize_map(mask,0.05,complement=True)
            #maskedin_img = mask*img
            #acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
            #acc_all = acc_all +acc
            
            #maskedin_img = np.transpose(maskedin_img.cpu().squeeze(dim=0).numpy(), (1, 2, 0))#maskedin_img.permute(1, 2, 0)#.numpy()
            

            
            #show_image_one(maskedin_img,std= config['training_opt']['std'],mean = config['training_opt']['mean'],dir =cls_dir+"masked"+str(j)+".png")
            #print(mask)
            #print(mask.shape)
            #exit()
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_generated.png', mask)
            #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/mask_gt.pSng', gt_mask)
            plt.imsave(cls_dir+"mask"+str(j)+".png", mask)
            #plt.imsave(cls_dir+"mask_GT"+str(j)+".png", gt_mask)


            binarized_mask = mask > 0.6
            intersection = np.logical_and(binarized_mask, gt_mask)
            union = np.logical_or(binarized_mask, gt_mask)
            iou = np.sum(intersection) / np.sum(union)
            IOU_list.append(iou)
            '''
            '''
            #evaluator.accumulate(mask,reference_mask.squeeze(dim=0).squeeze(dim=0).cpu().numpy())
            #if j ==99:
            #   print(total_time/100)
            #   exit()
    #all_masks = torch.stack(masks)
    #torch.save(all_masks, 'ScoreCam_masks_im_noise.pt')
    
    auc = calculate_auc(IOU_list,1/20000)
    print(auc)
    #auc= evaluator.compute()
    #print(auc)
    #acc_all = acc_all/count
    #print(acc_all)
    exit()





@torch.no_grad()
def eval_CAM_NICO(config, args, net):

    
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()


    maskin_saveroot = "/home/xzz5508/code/baseline_gallory/cam"


    gt_masks = read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    confidence_list = [i / 40.0 for i in range(1, 40)]
    evaluator =MaskEvaluator(confidence_list)

    
    image, label, context = load_NICO("test")
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    
    #testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=True)

    testing_dataset = get_test_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size= 1,
        shuffle=False
    )

    bb_model = torchvision.models.resnet18(pretrained=config['pretrained'])
    bb_model.fc = torch.nn.Linear(
        in_features=bb_model.fc.in_features, out_features=config['class_num'])
    bbweights_path = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_nico_2/resnet18-173-best.pth" 
    #"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/nico_resnet18/resnet18-127-best.pth"  # "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth" # "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/resnet18_imagenet_dropout/resnet18-71-best.pth"#"/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/imagenet9_resnet18/resnet18-193-best.pth"
    bb_model.load_state_dict(torch.load(bbweights_path))
    bb_model = bb_model.cuda()
    bb_model.eval()
    print(len(image))
    #print(len(all_data[0]))
    #print((all_data[0]))
    #exit()
    count = 0 
    acc_all = 0
    total_time = 0
    masks = []
    loaded_masks = torch.load('/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/score_Cam_masks_nico3.pt')
    # /home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/ScoreCam_masks_nico.pt

    for (img, label, context) in testing_dataset: 
        img = img.cuda()#.unsqueeze(dim=0)
        #print(img.permute(1, 2, 0).numpy().shape)
        #plt.imsave("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO"+'/img.png', img.permute(1, 2, 0).numpy())
        #show_image()
            
        #img = add_random_mask_pytorch(img)
        #mask = loaded_masks[count]
        count = count +1
        #time_start = time.time()
        #mask= ScoreCAM_generation(net,img)
        mask = CAM_mapgeneration(net,img)
        #masks.append(mask)
        #time_end = time.time()
        #time_used = time_end-time_start
        #total_time = total_time +time_used

        #print(mask.shape)
        #exit()
        #mask =mask.squeeze(dim=0).squeeze(dim=0)
        #mask = mask.cpu().numpy()

        '''
        '''
        mask = binarize_map(mask,0.05,complement=True)
        maskedin_img = mask*img
        acc = eval_postacc(maskedin_img,bb_model,torch.tensor([label]).cuda())
        acc_all = acc_all +acc
    #all_masks = torch.stack(masks)
    #torch.save(all_masks, 'Cam_masks_nico3.pt')
    #auc= evaluator.compute()
    #print(auc)
    acc_all = acc_all/count
    print(acc_all)
    exit()




#        _, preds = outputs.max(1)
#        correct += preds.eq(labels).sum()

def eval_postacc(masked_samples,bb_model,labels,COMET=False):
    #preds = bb_model(masked_samples)
    if COMET:
        preds,mask,comp =bb_model.forward_all(masked_samples)
    else:
        #preds = bb_model.training_step(masked_samples,labels,0)
        preds = bb_model(masked_samples)

    correct_num, acc = cal_acc(preds, labels)


    
    #train_correct += batch_correct
    #print(preds.shape)
    #print(train_acc)
    #exit()
    return correct_num

    






if __name__ == "__main__":
    pass