# train.py
#!/usr/bin/env	python3

import os
import random
#debug
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import matplotlib.pyplot as plt
import sys
import argparse
import time
import yaml
from datetime import datetime
from torch.autograd import Variable
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)
from conf import settings
from models.Saliency_mapper import Posthoc_loss
from utils import get_network, get_test_dataloader, get_val_dataloader, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
    update, get_mean_std, Acc_Per_Context, Acc_Per_Context_Class, penalty, cal_acc, get_custom_network, get_custom_network_vit, \
    save_model, load_model, get_parameter_number, init_training_dataloader,show_image
from train_module import train_env_ours, auto_split, refine_split, update_pre_optimizer, update_pre_optimizer_vit, update_bias_optimizer, auto_cluster
from eval_module import eval_training, eval_best, eval_mode,eval_explain,posthoc_eval,eval_CAM,posthoc_explain,B_cos_attribution,RBF_eval,eval_explain_NICO,posthoc_explain_NICO,eval_CAM_NICO,RBF_eval_nico,B_cos_attribution_nico
from timm.scheduler import create_scheduler
from CAM import CAM_mapgeneration
#from distillation import distill



def entropy(model_output):
    probabilities = F.softmax(model_output, dim=1)
    #print(probabilities.shape)
    # Calculate entropy using the formula
    #print((-torch.sum(probabilities * torch.log2(probabilities),dim=1)).shape)
    #exit()
    entropy = torch.mean(-torch.sum(probabilities * torch.log2(probabilities),dim=1))
    return entropy



def batch_total_variation_loss(masks):
    # masks should be of shape [batch_size, 1, height, width]
    pixel_dif1 = masks[:, :, 1:, :] - masks[:, :, :-1, :]  # Vertical difference
    pixel_dif2 = masks[:, :, :, 1:] - masks[:, :, :, :-1]  # Horizontal difference
    tv_loss = torch.sum(torch.abs(pixel_dif1), dim=[1, 2, 3]) + torch.sum(torch.abs(pixel_dif2), dim=[1, 2, 3])
    return tv_loss.mean()



def train(epoch):
    
    start = time.time()
    net.train()
    train_correct = 0.
    num_updates = epoch * len(train_loader)


    for batch_index, (images, labels) in enumerate(train_loader):
        if 't2tvit' in args.net and training_opt['optim']['sched']=='cosine':
            lr_scheduler.step_update(num_updates=num_updates)
        else:
            if epoch <= training_opt['warm']:
                if "3p" in config['exp_name']:
                    warmup_scheduler[0].step()
                    warmup_scheduler[1].step()
                    warmup_scheduler[2].step()
                elif "2p" in config['exp_name']:
                    warmup_scheduler[0].step()
                    warmup_scheduler[1].step()
                else:
                    warmup_scheduler.step()
                
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        if "COMET" in config['exp_name']:
            outputs,complement_pred,mask = net(images)
            #clean_out_p = net.predictor(images)
            #clean_out_c = net.completement_pred(images)
            # 244 is the original one (mistake); 32 is for cifar
            #print(mask.shape)
            #exit()

            l1_mask =torch.sum(torch.max(torch.sum(mask,dim=[-1,-2])/(224.*224.)-training_opt["mask_th"],torch.tensor(0.0))) 
            l1_loss =training_opt["m_m1"]* l1_mask/len(images) # 0.001 andd 20%  100*
        elif "posthoc" in config['exp_name'] :
            mask = net(images,labels)
        elif "RBF" in config['exp_name'] :
            #iteration = iteration+1

            loss_dict = net.training_step(images,labels,num_updates)
            #net.predictor.eval()
            #outputs = net.predictor(images)

            loss = loss_dict['loss']
        elif "Diet" in config['exp_name']:
            mask = net(images,labels)
        else:
            #batch_masks = torch.randint(0, 2, (224,224)).float()
            #images = images*batch_masks.cuda()
            outputs = net(images)
        
    

        # enviroment ----------------- repeat
        #labels = torch.cat((labels, labels), dim=0)
        #outputs = outputs[:len(labels)]

        #_,mask = net.forward_all(images)
        # _ : torch.Size([128, 60])
        # mask : torch.Size([128, 196, 1])
        # print(outputs2.size())

        #print(mask.size())
        #exit()
        

        if 'mixup' in training_opt and training_opt['mixup'] == True:
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
        else:
            if "COMET" in config['exp_name']:
                if "3p" in config['exp_name']:
                    complement = True
                    #TV_mask = 0.001*batch_total_variation_loss(mask)
                    TV_mask = 0
                    #loss_cleanp = loss_function(clean_out_p, labels)
                    #loss_cleanc = loss_function(clean_out_c, labels)
                    loss1 = loss_function(outputs, labels)
                    loss2 = loss_function(complement_pred,labels)
                    #entro_loss = entropy(complement_pred)
                
                    #gap_loss = torch.max(loss1 - loss2 ,torch.tensor(0.0))
                    #gap_loss = - loss2
                    if complement == True:
                        #smoothness = 
                        loss3 = training_opt["m_ploss"]*loss1 + training_opt["m_closs"]*(torch.max(loss1-loss2+training_opt["gap_th"],torch.tensor(0.0))) + l1_loss #+ TV_mask
                        #loss3 = training_opt["m_ploss"]*loss1 - training_opt["m_closs"]*loss2 + l1_loss
                        #loss3 = training_opt["m_ploss"]*loss1 -training_opt["m_closs"]*entro_loss + l1_loss
                    else:
                        loss3 = loss1 +l1_loss

                    optimizers[2].zero_grad()
                    loss3.backward(retain_graph=True)
                    net.generator.requires_grad_(False)
                

                    optimizers[0].zero_grad()
                    (loss1).backward(retain_graph=True)
                    net.predictor.requires_grad_(False)

                
                    optimizers[1].zero_grad()
                    (loss2).backward() #0.1*loss2+0.9*loss_cleanc

                    net.generator.requires_grad_(True)
                    net.predictor.requires_grad_(True)


                    optimizers[0].step()
                    optimizers[1].step()
                    optimizers[2].step()  # the generator is updated

                

                    loss = loss1
                elif "2p" in config['exp_name']:
                    complement = True
                    num_class = config["class_num"]
                    
                    #uniform_labels =  torch.full((len(labels), num_class), 1.0 / num_class).to(complement_pred.device)
                    
                    loss1 = loss_function(outputs, labels)
                    #complement_pred_dis = F.log_softmax(complement_pred,dim=-1)
                    #loss2 = kl_div_loss(complement_pred_dis,uniform_labels)
                    loss2 = loss_function(complement_pred, labels)
                    #entro_loss = entropy(complement_pred)
                    
                    
                    loss = training_opt["m_ploss"]*loss1 + training_opt["m_closs"]*(torch.max(loss1-loss2+training_opt["gap_th"],torch.tensor(0.0)))  +l1_loss
                    #loss = training_opt["m_ploss"]*loss1 -training_opt["m_closs"]*entro_loss + l1_loss
                    #loss_comp = loss_function(complement_pred, labels)
                    optimizers[0].zero_grad()
                    loss1.backward(retain_graph=True)
                    net.predictor.requires_grad_(False)
                
                    optimizers[1].zero_grad()
                    loss.backward()
                    net.predictor.requires_grad_(True)

                    optimizers[0].step()
                    optimizers[1].step()
                    #loss = loss1

                else:
                    complement = True
                    #num_class = config["class_num"]
                    #uniform_labels =  torch.full((len(labels), num_class), 1.0 / num_class).to(complement_pred.device)
                    loss1 = loss_function(outputs, labels)
                    loss2 = loss_function(complement_pred,labels)
                    #complement_pred_dis = F.log_softmax(complement_pred,dim=-1)
                    #loss2 = kl_div_loss(complement_pred_dis,uniform_labels)   
                    loss = loss1+training_opt["m_closs"]*(torch.max(loss1-loss2+training_opt["gap_th"],torch.tensor(0.0))) +l1_loss
                    #loss = training_opt["m_ploss"]*loss1 -training_opt["m_closs"]*entro_loss + l1_loss
                    #gap_loss = torch.max(loss1 - loss2 ,torch.tensor(0.0))
                    #gap_loss = loss1 - loss2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if batch_index ==0:
        
                    dir = "/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/" + exp_name +"/train_set"  +str(batch_index)+"/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    for j,image_show in enumerate(images):
                        one_set = dir+"_"+str(j)+"_"+str(labels[j])
                        # "number"+ str(j)+
                        image_show  = image_show.permute(1, 2, 0).cpu().numpy()
                        one_mask = mask[j].permute(1, 2, 0).cpu().detach().numpy()

                        #scaled_mask = upscale_mask(one_mask)
                        #scaled_mask = np.stack((scaled_mask,) * 3, axis=-1)
                        masked_image =  image_show*one_mask
                        one_mask = np.squeeze(one_mask)
                        show_image(masked_image,one_mask,image_show,one_set,std= config['training_opt']['std'],mean = config['training_opt']['mean'])

            elif "posthoc" in config['exp_name']:
                loss = loss_function.get(mask,images,labels.cuda(),explaindum)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif "RBF" in config['exp_name']:
                pass
            else:
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #loss_fullimg = loss_function(outputs_nomask, labels)

        #Loss = loss + 
        

       
        if "posthoc"  in config['exp_name']:
            train_acc = 0
        elif "RBF" in config['exp_name']:
            train_acc = loss_dict['acc']
            print(train_acc)
            
        else:
            batch_correct, train_acc = cal_acc(outputs, labels)
            train_correct += batch_correct

        num_updates += 1

        if "3p" in config['exp_name'] or "2p" in config['exp_name']:
            lr = optimizers[0].param_groups[0]['lr']
        elif "RBF" in config['exp_name']:
            lr = 0.0001
        else:
            lr = optimizer.param_groups[0]['lr']

        if batch_index % training_opt['print_batch'] == 0:
            if "3p" in config['exp_name']:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss_predictor: {:0.4f} \tLoss_complement: {:0.4f} \tLoss_Generator: {:0.4f} \tLR: {:0.6f}\tAcc: {:0.4f} \tL1 loss: {:0.4f} \tTV loss: {:0.4f} '.format(
                loss.item(),
                loss2.item(),
                loss3.item(),
                lr,
                train_acc,
                l1_loss,
                TV_mask,
                epoch=epoch,
                trained_samples=batch_index * training_opt['batch_size'] + len(images),
                total_samples=len(train_loader.dataset)
            ))
            elif "COMET" in config['exp_name']:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss_predictor: {:0.4f} \tLoss_uniform: {:0.4f} \tLoss_all: {:0.4f} \tLR: {:0.6f}\tAcc: {:0.4f} \tL1 loss: {:0.4f} '.format(
                loss1.item(),
                loss2.item(),
                loss.item(),
                lr,
                train_acc,
                l1_loss,
                epoch=epoch,
                trained_samples=batch_index * training_opt['batch_size'] + len(images),
                total_samples=len(train_loader.dataset)
            ))
            else:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tAcc: {:0.4f} '.format(
                    loss.item(),
                    lr,
                    train_acc,
                    epoch=epoch,
                    trained_samples=batch_index * training_opt['batch_size'] + len(images),
                    total_samples=len(train_loader.dataset)
            ))

    finish = time.time()
    train_acc_all = train_correct / len(train_loader.dataset)

    print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))
    return train_acc_all


torch.cuda.empty_cache()
if __name__ == '__main__':
    iteration =0
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='load the config file')
    parser.add_argument('-net', type=str, default='resnet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-multigpu', action='store_true', default=False, help='use multigpu or not')
    parser.add_argument('-name', type=str, default=None, help='experiment name')
    parser.add_argument('-debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('-eval', type=str, default=None, help='the model want to eval')
    parser.add_argument('-add_noise', type=bool, default=False, help='add noise to eval')
    # parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    
    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    args.net = config['net']
    # args.debug = False
    training_opt = config['training_opt']
    variance_opt = config['variance_opt']
    exp_name = args.name if args.name is not None else config['exp_name']



    



    if 'mixup' in training_opt and training_opt['mixup'] == True:
        print('use mixup ...')
    # ============================================================================
    # SEED
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(training_opt['seed'])
    if if_cuda:
        torch.cuda.manual_seed(training_opt['seed'])
        torch.cuda.manual_seed_all(training_opt['seed'])
    



    random.seed(training_opt['seed'])
    np.random.seed(training_opt['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ============================================================================
    # MODEL
    # debug: else
    if variance_opt['mode'] in ['ours']:
        if config['net'] == 'vit':
            net = get_custom_network_vit(args, variance_opt)
        else:
            net = get_custom_network(args, variance_opt)
    else:
        net = get_network(args,config)
    


    
    if "COMET"  in config["exp_name"] and args.eval is not None:
        
        #pass
        
        if config["dataset"] == "Imagenet9":
            print("imagenet9 test")
            eval_explain(config, args, net, None, None, args.eval)
        else:
            print("nico test")
            eval_explain_NICO(config, args, net, None, None, args.eval)
        exit()
        
    '''
    '''

    if "explaindum" in config and config["explaindum"] != None:
        explaindum = torchvision.models.resnet18(pretrained=False).cuda()
        explaindum.fc = torch.nn.Linear(
                in_features=explaindum.fc.in_features, out_features=config['class_num']).cuda()
        load_model(explaindum,config["explaindum"])
        #print(explaindum.__dict__)
        #exit()
    else:
        explaindum = None

    if "CAM" in config["exp_name"]:
        eval_CAM(config, args,explaindum)
        #eval_CAM_NICO(config, args,explaindum)
        exit()
    # ============================================================================
    # DATA PREPROCESSING
    mean, std = training_opt['mean'], training_opt['std']

    train_loader_init = init_training_dataloader(config, mean, std, variance_opt['balance_factor'])

    # false in debugging
    pre_split = None



    val_loader = get_val_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )


    test_loader = get_test_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )


    if "explaindum" not in config :
        loss_function = nn.CrossEntropyLoss()
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    
    
    


    if args.eval is not None:

        #loss_function = nn.CrossEntropyLoss() # used only for test posthoc-acc
        eval_explain(config, args, net, val_loader, loss_function, args.eval)
        
        #eval_explain_NICO(config, args, net, val_loader, loss_function, args.eval)
        #posthoc_explain(config, args, net,explaindum,args.eval)
        #posthoc_explain_NICO(config, args, net,explaindum,args.eval)
        
        #output=net.eval_model(test_loader) 
        #print(output)
        #exit()
        #val_acc = eval_mode(config, args, net, val_loader, loss_function, args.eval,explaindum)
        #test_acc = eval_mode(config, args, net, test_loader, loss_function, args.eval,explaindum)
        #print(test_acc)
        #test_acc =  eval_best(config, args, net, test_loader, loss_function ,args.eval, 0)
        #print('Val Score: %s  Test Score: %s' %(val_acc.item(), test_acc.item()))
        exit()



            # false


    train_loader = train_loader_init.get_dataloader(training_opt['batch_size'], num_workers=4, shuffle=True)


    
    # true
    if 'COMET' in config['exp_name']:
        if "3p" in config['exp_name']:
            optimizer1 = optim.AdamW(net.predictor.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer2 = optim.AdamW(net.completement_pred.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer3 = optim.AdamW(net.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizers = [optimizer1,optimizer2,optimizer3]
        elif "2p" in config['exp_name']:
            optimizer1 = optim.AdamW(net.predictor.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer2 = optim.AdamW(net.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizers = [optimizer1,optimizer2]
        else:
            optimizer = optim.AdamW(net.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)
        # false

        if "3p" in config['exp_name'] or "2p" in config['exp_name']:
            train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2) for optimizer in optimizers]  # learning rate decay
            iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
            warmup_scheduler = [WarmUpLR(optimizer, iter_per_epoch * training_opt['warm']) for optimizer in optimizers]
        else:
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2) 
            iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * training_opt['warm'])
    

    #optimizer = optim.SGD(net.parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4)
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2)  # learning rate decay
    #iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * training_opt['warm'])    


    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, exp_name)

    if args.debug:
        checkpoint_path = os.path.join(checkpoint_path, 'debug')




    #create checkpoint folder to save model    
    checkpoint_path = "/home/xzz5508/code/Imbalance_ood/ECCV_camera_ready/COMET/"
    
    if "posthoc" in exp_name:
        checkpoint_path = checkpoint_path+"posthoc/"
    checkpoint_path = checkpoint_path+exp_name

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open(checkpoint_path+"/config.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #net.selector.load_state_dict(torch.load("/home/xzz5508/code/Imbalance_ood/Our_basedon_caam/0-NICO/RBF_trainselector_resnet18_2/RBF_trainselector-38-best.pth"))
    #use tensorboard
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    writer = SummaryWriter(log_dir=checkpoint_path)

    best_acc = 0.
    if "posthoc" in exp_name or "RBF" in exp_name:
        best_acc = 100
    best_epoch = 0
    best_train_acc = 0.
    if 'pretrain' in config and config['pretrain'] is not None:
        state_dict = torch.load(config['pretrain'])
        net.load_state_dict(state_dict, strict=False)
        print('Loaded pretrained model...')
    

    for epoch in range(1, training_opt['epoch']):
        if 't2tvit' in args.net and training_opt['optim']['sched']=='cosine':
            lr_scheduler.step(epoch)
        else:
            if epoch > training_opt['warm']:
                if "RBF" in config['exp_name']:
                    pass
                elif isinstance(train_scheduler, list):
                    for train_scheduler_ in train_scheduler:
                        train_scheduler_.step()
                else:
                    train_scheduler.step()
        if config['resume']:
            if epoch <= resume_epoch:
                continue


        train_acc = train(epoch)

        # for posthoc and RBF, loss is 'acc'. smaller better
        if "posthoc" in exp_name:
            acc = posthoc_eval(config, args, net, explaindum,val_loader, loss_function, writer, epoch)
            if best_acc > acc:
                # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                best_train_acc  = train_acc
        elif "RBF" in exp_name:
            #acc = posthoc_eval(config, args, net, explaindum,val_loader, loss_function, writer, epoch)
            results = net.eval_model(val_loader)
            acc  = results["loss"]
            print("-RBF----loss----kl-----acc")
            print(results)
            if best_acc > acc:
                # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                
                if "trainselector"  in exp_name:
                    print("save selector")
                    save_model(net.selector, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                else:
                    save_model(net.predictor, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                best_train_acc  = train_acc

        else:
            acc = eval_training(config, args, net, val_loader, loss_function, writer, epoch)
            if best_acc < acc:
                # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                best_train_acc  = train_acc
        

    
        if not epoch % training_opt['save_epoch']:
            # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            if "trainselector" in exp_name:
                save_model(net.selector, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            elif "trainpredictor" in exp_name:
                save_model(net.predictor, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

            else:
                save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        print("Best Acc: %.4f \t Train Acc: %.4f \t Best Epoch: %d" %(best_acc, best_train_acc, best_epoch))

    print('Evaluate Best Epoch %d ...' %(best_epoch))
    if "posthoc" in exp_name:
        acc_final = posthoc_eval(config, args, net,explaindum ,test_loader, loss_function, writer, epoch)
    else:
        acc_final = eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch)
    txt_write = open("results_txt/" + exp_name + '.txt', 'w')
    txt_write.write(str(best_train_acc.cpu().item()))
    txt_write.write(str(best_acc.cpu().item()))
    txt_write.write(str(acc_final.cpu().item()))
    txt_write.close
    writer.close()
