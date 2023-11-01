# -*- coding: gbk -*-
from scipy.spatial.distance import cosine
import torch
import sys
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from vit_model import vit_base_patch16_224_in21k as create_model
import time
import PIL.Image as Image
import glob
import numpy as np
from torchvision import transforms
from ContrastLoss import Contrast_Losses,infoNCE_loss
from tqdm import tqdm
from torchvision import datasets, transforms
import os
import math
import argparse
import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_data2list(path):
        personal_name_lst=[]
        personal_data_lst=[]
        camera_list=[]
        with open(path,'r',encoding='utf-8') as f:
            for info in f.readlines():
                contant = info.strip('\n').split('\t')
                name = contant[0]
                camera=contant[0].split('-')[-2]
                data = [float(x) for x in contant[-1].split('<=>')]
                personal_name_lst.append(name)
                personal_data_lst.append(data)
                camera_list.append(camera)
        f.close()
        return personal_name_lst,personal_data_lst,camera_list

def get_code(angle,num_images_per_folder,epoch): 
             
    batch_size=1024*num_images_per_folder

    train_encode_path1='./train_{}encode/{}/train'.format(num_images_per_folder,angle)
    
    personal_name_lst1,personal_data_lst1,_=load_data2list(train_encode_path1)   

    
    target_lst = personal_name_lst1[::num_images_per_folder]  
      
    train_data=torch.tensor(personal_data_lst1,dtype=torch.float)
    test_train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False)
    

    save_encode='./result_code/{}'.format(angle)
    os.makedirs(save_encode,exist_ok=True)
    
    model_ = torch.load('./ckpt_12440/{}/model_{}.pth'.format(angle,epoch)).to(device)
    model_.eval()
     
    with torch.no_grad():      
        out = []
        for x1 in test_train_dataloader:
            x1 = x1.to(device)
            outputs = model_(x1)
            out.extend(outputs)
        with open(save_encode+'/encode_{}.txt'.format(epoch),'w',encoding="utf-8")as f:
            for name,en in tqdm(zip(target_lst,out)):
                en = en.tolist()
                en = [str(x)for x in en]
                en = '<=>'.join(en)
                s = '\t'.join([name,en])
                f.write(s+'\n')
        del out
    code_path=save_encode+'/encode_{}.txt'.format(epoch)
    return code_path
#################################################################################################
def trans_encod(encode):
    ret=[float(x)for x in encode.split("<=>")]
    return ret

def load_data(input_path):
    path=input_path
    t_cam2person={}
    
    for line in open(path,"r",encoding="utf-8"):#=================
        idd,encode=line.strip().split("\t")
        person_id,cond,cam_id,u1,u2=idd.split("-")
        if cam_id not in t_cam2person:
            t_cam2person[cam_id]={}
        t_cam2person[cam_id][person_id]=trans_encod(encode)
    return t_cam2person

def compare(prob,galary,f_name):
    first=[]
    second=[]
    with open("./pair_detail/"+f_name,"w",encoding="utf-8") as f:
        for p_id,encode_p in prob.items():
            this_p2dis=[]
            if p_id not in galary:
                this_p2dis.append([-0.5,p_id])
            for g_id,encode_g in galary.items():
                get_dis= cosine(encode_p, encode_g)
                this_p2dis.append([get_dis,g_id])
            this_p2dis.sort()
            first.append(this_p2dis[0][0])
            second.append(this_p2dis[1][0])
            to_save="%s=>"%p_id
            for dis,person in this_p2dis:
                to_save +="%s->%s||"%(person,dis)
            f.write(to_save+"\n")
    return first,second

def plot_from_result():
    file_lst=glob.glob("./pair_detail/*")
    first=[]
    second=[]
    for file_ in file_lst:
        for line in  open(file_,"r",encoding="utf-8"):
            p,others=line.strip().split("=>")
            others_lst=others.split("||")
            this_f,this_s=others_lst[:2]
            this_f=float(this_f.split("->")[1])
            this_s=float(this_s.split("->")[1])
            first.append(this_f)
            second.append(this_s)
    return first,second
    
def get_acc():
    path = glob.glob('./pair_detail/*')
    lst_1 = []
    for line in path:
        y=0
        u = 0
        for i in open(line,'r',encoding="utf-8"):
            ii = i.strip().split('->')
            ii_ = ii[0].split('=>')
            u+=1
            if ii_[0]==ii_[1]:
                y+=1
        acc = (y/u)*100
        lst_1.append(acc)
    mean = np.mean(lst_1)
    return mean    
def get_result(angle,num_images_per_folder,epoch):
 
    input_path=get_code(angle,num_images_per_folder,epoch) 
          
    start_over=True
    if start_over:
        t_cam2person=load_data(input_path)
        cam_id_lst=[x[0] for x in t_cam2person.items()]
        first=[]
        second=[]
        for i in range(len(cam_id_lst)):
            for j in range(len(cam_id_lst)):
                if i == j: continue
                probe_cam=cam_id_lst[i]
                galary_cam=cam_id_lst[j]
                probe=t_cam2person[probe_cam]
                galary=t_cam2person[galary_cam]
                f_name="%svs%s.txt"%(probe_cam,galary_cam)

                this_first,this_second=compare(probe,galary,f_name)
                first +=this_first
                second +=this_second
    else:
        first,second=plot_from_result()
    top1=get_acc()
      
    return top1
                    
if __name__ == '__main__':
    #path=get_code(angle,num_images_per_folder,epoch)
    angle=0
    num_images_per_folder=40
    epoch=5000
    get_result(angle,num_images_per_folder,epoch)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
