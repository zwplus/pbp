import torch
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import random
import json
import cv2
from transformers import CLIPImageProcessor

class diffusion_dataset(Dataset):
    def __init__(self,data_pairs_txt_list) -> None:
        super().__init__()
        self.data_pairs=[]
        pairs_list=[]
        for i in data_pairs_txt_list:
            with open(i,'r') as f:
                pairs_list.extend(f.readlines())
        
        # for i in pairs_list:
        #     i=i.strip()
        #     ref_img_path,people_img_path,back_img_path,local_img_dir=i.split(',')
        #     #获取图片名称，因为图片的文件名代表着相应各个身体部位所在的文件夹
        #     ref_local_path=local_img_dir
        #     self.data_pairs.extend([(ref_img_path,people_img_path,back_img_path,ref_local_path,) ])
        for i in pairs_list:
            i=i.strip()
            people_img_path,local_img_dir=i.split(',')
            #获取图片名称，因为图片的文件名代表着相应各个身体部位所在的文件夹
            ref_local_path=local_img_dir
            self.data_pairs.extend([(people_img_path,ref_local_path) ])
        print(len(pairs_list))
        self.transformer_ae=transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        # self.transformer_style_local=transforms.Compose(
        #     [
        #     transforms.ToTensor(),
        #     transforms.Resize((256,256)),
        #     torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     ]
        # )

        self.transformer_clip=CLIPImageProcessor.from_pretrained('/root/data1/github/pbp/sd-image-variations-diffusers/feature_extract')

        # if if_test==True:
        #     random.shuffle(self.data_pairs)
        #     self.data_pairs=self.data_pairs[:160]

    def __len__(self,):
        return len(self.data_pairs)

    # def __getitem__(self, index):
    #     ref_path,ref_people,ref_back,ref_local_path=self.data_pairs[index]
    #     target_img_path=ref_path
    #     image=Image.open(target_img_path)
    #     target_img=self.transformer_ae(image)

    #     ref_people_img=Image.open(ref_people)
    #     ref_people_img=self.transformer_style(ref_people_img)

    #     ref_local_img=[]
    #     for i in os.listdir(ref_local_path):
    #         ref_local_img.append(self.transformer_style_local(Image.open(os.path.join(ref_local_path,i))))
    #     ref_local_img=torch.cat(ref_local_img,dim=0)

    #     ref_back_image=Image.open(ref_back)
    #     ref_back=self.transformer_style(ref_back_image)
    #     ref_back_image=self.transformer_ae(ref_back_image)

    #     return target_img,ref_people_img.pixel_values,ref_local_img,ref_back_image,ref_back.pixel_values

    def __getitem__(self, index):
        ref_people,ref_local_path=self.data_pairs[index]
        image=Image.open(ref_people)

        target_img=self.transformer_ae(image)
        people_img=self.transformer_clip(image)

        ref_local_img=[]
        for i in os.listdir(ref_local_path):
            ref_local_img.append(self.transformer_ae(Image.open(os.path.join(ref_local_path,i))))
        ref_local_img=torch.cat(ref_local_img,dim=0)


        return target_img,people_img.pixel_values,ref_local_img