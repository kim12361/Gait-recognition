# -*- coding: gbk -*-
import timm
import torch
from PIL import Image
from torchvision import transforms# Load ViT 
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import shutil
import glob
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from vit_encode_model import vit_base_patch16_224_in21k as create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device='cpu'

for i in range(1):
    model = create_model().to(device)
    weights = 'vit_base_patch16_224_in21k.pth'
    if weights != "":
        assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
        weights_dict = torch.load(weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
    freeze_layers = True

    if freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

#model = timm.create_model('convnext_xlarge_in22k', pretrained=True, num_classes=0)
#weight=

def load_data_62(source_folder,image_size):
    
    #image_size = (224, 224) # 目标图片大小
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    subfolders= random.sample(subfolders, 62)
    
    
    tensorlist = []
    target_lst = []
    random_tensorlist = []
    random_target_lst = []
    for folder in tqdm(subfolders):
        contents = [f.path for f in os.scandir(folder)]
        contents = [contents[0],contents[2],contents[5],contents[6],contents[8],contents[9]]
        for f in contents:
            image_files = glob.glob(f + '/*')
            image_files = image_files[ang]
            print(image_files)
            #exit()
            image_files = [f.path for f in os.scandir(image_files)]
            #print(image_files)
            #exit()
            if len(image_files) < num_images_per_folder:
                continue
            if len(image_files) >= num_images_per_folder:
                selected_images = image_files[:num_images_per_folder]
                
            random_selected_images = random.sample(image_files,num_images_per_folder)
            x_lst = [] 
            random_x_list=[]
            for image_path in selected_images:
                y = image_path.split('/')[-1].split('.')[0]
                target_lst.append(y)
                with Image.open(image_path) as img:
                    img = img.convert("RGB") # 转换为RGB格式
                    img = img.resize((image_size), Image.ANTIALIAS) # 调整大小
                    re_img = np.asarray(img)/255.0
                    x_lst.append(re_img)
            x_tensor = torch.Tensor(x_lst)
            tensorlist.append(x_tensor)
            
            for random_image_path in random_selected_images:
                y = random_image_path.split('/')[-1].split('.')[0]
                random_target_lst.append(y)
                with Image.open(random_image_path) as img1:
                    img1 = img1.convert("RGB") # 转换为RGB格式
                    img1 = img1.resize((image_size), Image.ANTIALIAS) # 调整大小
                    re_img1 = np.asarray(img1)/255.0
                    random_x_list.append(re_img1)
            random_x_tensor = torch.Tensor(random_x_list)
            random_tensorlist.append(random_x_tensor)
            
    x_tensor = torch.cat(tensorlist, dim=0)
    random_x_tensor = torch.cat(random_tensorlist, dim=0)
    return x_tensor,target_lst,random_x_tensor,random_target_lst
    
    
def load_data(source_folder,image_size):#正样本
    
    #image_size = (224, 224) # 目标图片大小
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    #subfolders=subfolders[:1]
    #random.shuffle(subfolders)
    print(len(subfolders),'subfolders')
    tensorlist = []
    target_lst = []
    for folder in tqdm(subfolders):
        #folder = [0:]
        contents = [f.path for f in os.scandir(folder)]
        
        contents = [contents[0],contents[2],contents[5],contents[6],contents[8],contents[9]]
        
        for f in contents:
            image_files = glob.glob(f + '/*')
            image_files = image_files[ang]
            print(image_files)
            #exit()
            image_files = [f.path for f in os.scandir(image_files)]
            #print(image_files)
            #exit()
            if len(image_files) < num_images_per_folder:
                continue
            
            #image_files = sorted(image_files)
            selected_images = image_files[:num_images_per_folder]
            #selected_images = random.sample(image_files, num_images_per_folder)
            x_lst = []

            for image_path in selected_images:
                y = image_path.split('/')[-1].split('.')[0]
            
                target_lst.append(y)
                
                with Image.open(image_path) as img:
                    img = img.convert("RGB") # 转换为RGB格式
                    img = img.resize((image_size), Image.ANTIALIAS) # 调整大小
                    re_img = np.asarray(img)/255.0
                    x_lst.append(re_img)
            x_tensor = torch.Tensor(x_lst)
            
            tensorlist.append(x_tensor)
    #target_lst = target_lst[0:len(target_lst):num_images_per_folder]
    
    x_tensor = torch.cat(tensorlist, dim=0)
    print(len(target_lst))
    print(x_tensor.shape)
    return x_tensor,target_lst

def load_data1(source_folder,image_size):#负样本
    
    #image_size = (224, 224) # 目标图片大小
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    #subfolders=subfolders[0:30]

    tensorlist = []
    target_lst = []
    for folder in tqdm(subfolders):
        #folder = [0:]
        contents = [f.path for f in os.scandir(folder)]
        contents = [contents[0],contents[2],contents[5],contents[6],contents[8],contents[9]]
        #print(contents)
        #exit()
        for f in contents:
            image_files = glob.glob(f + '/*')
            image_files = image_files[ang]#===============================================================角度==
            print(image_files)
            #exit()
            image_files = [f.path for f in os.scandir(image_files)]
            
            if len(image_files) < num_images_per_folder:
                continue

            selected_images = random.sample(image_files, num_images_per_folder)
            x_lst = []
            # target_lst = []
            for image_path in selected_images:
                y1 = image_path.split('/')[-1].split('.')[0]
                target_lst.append(y1)
                
                with Image.open(image_path) as img:
                    img = img.convert("RGB") # 转换为RGB格式
                    img = img.resize((image_size), Image.ANTIALIAS) # 调整大小
                    re_img = np.asarray(img)/255.0
                    x_lst.append(re_img)
            x_tensor = torch.Tensor(x_lst)
            
            tensorlist.append(x_tensor)

    x_tensor = torch.cat(tensorlist, dim=0)
    return x_tensor,target_lst

if __name__ == '__main__':
    
    '''
    0:'./data/001/nm-06/144', 
    1:'./data/001/nm-06/108', 
    2:'./data/001/nm-06/090', 
    3:'./data/001/nm-06/018', 
    4:'./data/001/nm-06/180', 
    5:'./data/001/nm-06/054', 
    6:'./data/001/nm-06/162', 
    7:'./data/001/nm-06/036', 
    8:'./data/001/nm-06/126', 
    9:'./data/001/nm-06/000', 
    10:'./data/001/nm-06/072'

    '''
    print(model)
    #test_source_folder_full="./62"
    train_source_folder="./train_data" 
    test_source_folder="./test_data" 
    
    num_images_per_folder = 40#每个子文件夹选取的图片数量
    
    angle_list=[144,108,90,18,180,54,162,36,126,0,72]
    
    for i1 in range(1):
        i=3
        ang=i#############################################序号
        angle_test=angle_list[i]#############################################角度
        
        encode_path='./train_{}encode/{}/'.format(num_images_per_folder,angle_test)
        os.makedirs(encode_path, exist_ok=True)
        batch_size=1
        '''
        x_tensor,target_lst,random_x_tensor,random_target_lst = load_data_62(train_source_folder,image_size=(224,224))
        
        train_dataloader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
        random_train_dataloader = DataLoader(random_x_tensor, batch_size=batch_size, shuffle=False)
        '''
        '''
        #负样本
        test_x_tensor1,target_lst1 = load_data1(train_source_folder,image_size=(224,224))#random
        random_train_dataloader = DataLoader(test_x_tensor1, batch_size=batch_size, shuffle=False)
        '''
        #正样本
        test_x_tensor2,target_lst2 = load_data(train_source_folder,image_size=(224,224))
        train_dataloader = DataLoader(test_x_tensor2, batch_size=batch_size, shuffle=False)
        '''
        #测试缺陷
        test_x_tensor4,target_lst4 = load_data(test_source_folder,image_size=(224,224))
        test_dataloader = DataLoader(test_x_tensor4, batch_size=batch_size, shuffle=False)
        '''
        n=0
        m=0
        
        with torch.no_grad():
            for i in range(1):
                out = []      
                for x1 in train_dataloader:
                    
                    x1 = x1.to(device)
                    x1 = x1.permute(0,3,1,2)
                    model=model.to(device)
                    outputs = model(x1)
                    print(outputs.shape)
                    m+=1
                    out.extend(outputs)
                print(m)
                with open(encode_path+'train','w',encoding="utf-8")as f:
                    
                    for name,en in tqdm(zip(target_lst2,out)):
                        
                        en = en.tolist()
                        en = [str(x)for x in en]
                        en = '<=>'.join(en)
                        s = '\t'.join([name,en])
                        f.write(s+'\n')
                        n+=1
                del out       
        '''     
        with torch.no_grad():
            for i in range(1):
                out = []      
                for x1 in random_train_dataloader:
                    
                    x1 = x1.to(device)
                    x1 = x1.permute(0,3,1,2)
                    model=model.to(device)
                    outputs = model(x1)
                    
                    out.extend(outputs)
                
                with open(encode_path+'train_random','w',encoding="utf-8")as f:
                    for name,en in tqdm(zip(target_lst1,out)):
                        
                        en = en.tolist()
                        en = [str(x)for x in en]
                        en = '<=>'.join(en)
                        s = '\t'.join([name,en])
                        f.write(s+'\n')
                    
                del out  
        '''
        '''
        with torch.no_grad():
            for i in range(1):
                out = []      
                for x1 in test_dataloader:
                    
                    x1 = x1.to(device)
                    x1 = x1.permute(0,3,1,2)
                    model=model.to(device)
                    outputs = model(x1)
                    n+=1
                    out.extend(outputs)
                print(n)
                with open(encode_path+'test','w',encoding="utf-8")as f:
                    for name,en in zip(target_lst4,out):
                        
                        en = en.tolist()
                        en = [str(x)for x in en]
                        en = '<=>'.join(en)
                        s = '\t'.join([name,en])
                        f.write(s+'\n')
                        
        '''
                    
                         
                            