
import os
import pickle
import json
import random
import numpy as np


def get_fgs(class_name):
        all_images = sorted(os.listdir(class_name))
        path_set = []
        for image in all_images:
            path_set.append(f'{class_name}/{image}')
        return path_set

def save_pickle(saved_file,address):
    
    file = open(address,"wb")

    pickle.dump(saved_file,file)

    file.close()

def save_json(saved_file,address):
    
    file = open(address,"w")

    json.dump(saved_file,file)

    file.close()


def cut_subset(data,num):
    new_dataset = []
    for image_class in data:
        random_samples = random.sample(image_class, num)
        new_dataset.append(random_samples)
    return new_dataset

def main():
    data_root = "/home/xzz5508/code/Imbalance_ood/Imagenet_9"
    original_data = f'{data_root}/original/train' 
    class_names = os.listdir(original_data)


    train_data_set = [[] for i in range(9)]
    valid_data_set = [[] for i in range(9)]
    test_data_set =[[] for i in range(9)]
    for i, class_name in  enumerate(class_names):
        train_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/original/val' 
    for i, class_name in  enumerate(class_names):
        valid_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/bg_challenge/original/val' 
    for i, class_name in  enumerate(class_names):
        test_data_set[i] = get_fgs(f'{original_data}/{class_name}')


    #train_data_subset = cut_subset(train_data_set,1200)
    #val_data_subset= cut_subset(valid_data_set,120)
    #test_data_subset= cut_subset(test_data_set,120)


    # save_pickle(train_data_set,address=data_root+"/Imagenet_original_train.pickle")
    # save_pickle(valid_data_set,address=data_root+"/Imagenet_original_val.pickle")
    # save_pickle(test_data_set,address=data_root+"/Imagenet_original_test.pickle")
    
    #print(train_data_set[0][0])
    labels_dic = {k: v for v, k in enumerate(class_names)}
    #save_json(labels_dic,data_root+"name2label.json")
    print(labels_dic)
    print(class_names)




def read_mask(mask_path,datapath_pickle):
    file = open(datapath_pickle, 'rb')
    data_addresses= pickle.load(file)
    file.close()

    for oneclass in data_addresses:
        class_name = oneclass[0].split("/")[-2]
        image_name = oneclass[0].split("/")[-1].split('.')[0]

        path = mask_path+"/"+class_name+"/"+image_name+".npy"
        image = np.load(path)

def get_fg_images():
    data_root = "/home/xzz5508/code/Imbalance_ood/Imagenet_9"
    original_data = f'{data_root}/only_fg/train' 
    class_names = os.listdir(original_data)


    train_data_set = [[] for i in range(9)]
    valid_data_set = [[] for i in range(9)]
    test_data_set =[[] for i in range(9)]
    for i, class_name in  enumerate(class_names):
        train_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/only_fg/val' 
    for i, class_name in  enumerate(class_names):
        valid_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/bg_challenge/only_fg/val' 
    for i, class_name in  enumerate(class_names):
        test_data_set[i] = get_fgs(f'{original_data}/{class_name}')
    

    save_pickle(train_data_set,address=data_root+"/Imagenet_only_fg_train.pickle")
    save_pickle(valid_data_set,address=data_root+"/Imagenet_only_fg_val.pickle")
    save_pickle(test_data_set,address=data_root+"/Imagenet_only_fg_test.pickle")

def get_mixed_next_images():
    data_root = "/home/xzz5508/code/Imbalance_ood/Imagenet_9"
    original_data = f'{data_root}/no_fg/train' 
    class_names = os.listdir(original_data)


    train_data_set = [[] for i in range(9)]
    valid_data_set = [[] for i in range(9)]
    test_data_set =[[] for i in range(9)]
    for i, class_name in  enumerate(class_names):
        train_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/no_fg/val' 
    for i, class_name in  enumerate(class_names):
        valid_data_set[i] = get_fgs(f'{original_data}/{class_name}')

    original_data = f'{data_root}/bg_challenge/no_fg/val' 
    for i, class_name in  enumerate(class_names):
        test_data_set[i] = get_fgs(f'{original_data}/{class_name}')
    

    save_pickle(train_data_set,address=data_root+"/Imagenet_no_fg_train.pickle")
    save_pickle(valid_data_set,address=data_root+"/Imagenet_no_fg_val.pickle")
    save_pickle(test_data_set,address=data_root+"/Imagenet_no_fg_test.pickle")

    #print(train_data_set[0][0])
    #labels_dic = {k: v for v, k in enumerate(class_names)}
    #print(labels_dic)
    #print(class_names)
    #save_json(labels_dic,data_root+"name2label.json")



def only_mask(file_addresses):
    mask_only = []
    for address in file_addresses:

        
        if "masked" in address:
            
            mask_only.append(address)

    return mask_only



def split_train_val_test(only_masked,train_num,val_num):
    train_set = only_masked[:train_num]
    val_set = only_masked[train_num:val_num+train_num]
    test_set = only_masked[val_num+train_num:]

    return train_set,val_set,test_set

def masked_dataset(dataset_root):

    train_data_set = [[] for i in range(9)]
    valid_data_set = [[] for i in range(9)]
    test_data_set =[[] for i in range(9)]




    for i in range(9):

        #class_names = os.listdir(dataset_root)
        #print(class_names)
        fgs = get_fgs(f'{dataset_root}/{i}')
        only_masked = only_mask(fgs)

        train_set,val_set,test_set = split_train_val_test(only_masked,350,50)
        train_data_set[i] = train_set
        valid_data_set[i] = val_set
        test_data_set[i] = test_set


    save_pickle(train_data_set,"/home/xzz5508/code/Imbalance_ood/Imagenet_9/Maskout_dataset/Rationale_change/maskout_train.pickle")
    save_pickle(valid_data_set,"/home/xzz5508/code/Imbalance_ood/Imagenet_9/Maskout_dataset/Rationale_change/maskout_val.pickle")
    save_pickle(test_data_set,"/home/xzz5508/code/Imbalance_ood/Imagenet_9/Maskout_dataset/Rationale_change/maskout_test.pickle")

    return 0








if __name__ == "__main__":
    #read_mask("/home/xzz5508/code/Imbalance_ood/Imagenet_9/bg_challenge/fg_mask/val","/home/xzz5508/code/Imbalance_ood/Imagenet_9/Imagenet_original_test.pickle")
    #main()
    #get_fg_images()
    get_mixed_next_images()

    #masked_dataset("/home/xzz5508/code/maskout_samples/Rationale_change")
