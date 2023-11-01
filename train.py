# -*- coding: gbk -*-
from test_result import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import net
import torch
import sys
import timm
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
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
#import matplotlib
#matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
''' 
num_gpus = torch.cuda.device_count()

print(f"Total available GPUs: {num_gpus}")

# 遍历每个GPU，显示序号和名称
for gpu_id in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"GPU {gpu_id}: {gpu_name}")
    
exit() 
'''                
def train_model(train_dataloader,train_power_dataloader,batch_size,lr,angle,images_per_folder1,n,all_epoch):

    model =net.SimCLRStage1(images_per_folder1).to(device)
    #model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    top1_list=[40]
    top2v4_list=[]
    epoch_list=[0]
    loss_list=[]
    
    for epoch in range(all_epoch):
        start_time=time.time()
        model.train()
        for x1,x2 in zip(train_dataloader,train_power_dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            z1 = model(x1)
            z2 = model(x2)
            print(z1.shape)
            loss = Contrast_Losses(z1, z2)

            loss_list.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time=time.time()
        #print("[epoch=={}]------Loss:{}----------spend_time:{}".format(epoch, loss.item(),end_time-start_time))
        
        
        if (epoch+1) %n== 0:
            ckpt_124="./ckpt_124{}/{}".format(images_per_folder1,angle)
            os.makedirs(ckpt_124,exist_ok=True)
            
            torch.save(model, ckpt_124+"/model_{}.pth".format(epoch+1))
            
            top1=get_result(angle,images_per_folder1,epoch+1)
            
            if top1>max(top1_list):
                epoch_list.append(epoch+1)
                
            print(top1,'top1_accuracy')
            top1_list.append(top1)
            
        print("[epoch=={}]-----Loss:{}-----angle:{} --spend_time:{}".format(epoch, loss.item(),angle,end_time-start_time))#,max(epoch_list),max(top1_list)))
        if (epoch+1) %n== 0:
            plt.figure()
            plt.plot(range(len(top1_list)),top1_list,c='r',label='epoch:{}-top1-{}'.format(max(epoch_list),max(top1_list)))
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            os.makedirs('./result_124{}/{}'.format(images_per_folder1,angle),exist_ok=True)
            plt.savefig('./result_124{}/{}/top1.png'.format(images_per_folder1,angle))
            
            plt.figure()
            plt.plot(range(len(loss_list)),loss_list,c='r',label='loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            os.makedirs('./result_124{}/{}'.format(images_per_folder1,angle),exist_ok=True)
            plt.savefig('./result_124{}/{}/loss.png'.format(images_per_folder1,angle))
            if max(top1_list)==100:
              exit()
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
        print(len(personal_name_lst))
        return personal_name_lst,personal_data_lst,camera_list




if __name__ == '__main__':
        #angle_list=[180]
        
        for ang1 in range(1):
            angle=90
            num_images_per_folder = 40
            print(num_images_per_folder,'num_images_per_folder')
            
            all_epoch=100000
            num_test=200
            lr = 0.0001
            batch_size=1024*num_images_per_folder
            
            train_encode_path='./train_{}encode/{}/train'.format(num_images_per_folder,angle)
            train_rand_encode_path='./train_{}encode/{}/train_random'.format(num_images_per_folder,angle)
            
            personal_name_lst,personal_data_lst,_=load_data2list(train_encode_path)
            personal_name_lst1,personal_data_lst1,_=load_data2list(train_rand_encode_path)


            #####################
            train_data=torch.tensor(personal_data_lst,dtype=torch.float)
            train_random_data=torch.tensor(personal_data_lst1,dtype=torch.float)
            
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
            random_train_dataloader = DataLoader(train_random_data, batch_size=batch_size,shuffle=False)
            
            train_model(train_dataloader,random_train_dataloader,batch_size,lr,angle,num_images_per_folder,num_test,all_epoch)
            