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
    def __init__(self,data_pairs_txt_list,if_train=True) -> None:
        super().__init__()
        self.data_pairs=[]
        pairs_list=[]
        for i in data_pairs_txt_list:
            with open(i,'r') as f:
                pairs_list.extend(f.readlines())
        
        for i in pairs_list:
            i=i.strip()
            back_image_path,local_img_dir,pose_img_path,target_img_path=i.split(',')
            if not os.path.isdir(local_img_dir):
                print(local_img_dir)
            #获取图片名称，因为图片的文件名代表着相应各个身体部位所在的文件夹
            ref_local_path=local_img_dir
            people=os.path.join('/'.join(local_img_dir.split('/')[:-4]),'groundsam_people_img',local_img_dir.split('/')[-2],local_img_dir.split('/')[-1]+'.png')
            self.data_pairs.extend([(back_image_path,ref_local_path,people,pose_img_path,target_img_path) ])
        
        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        min_crop_scale = 0.8 if if_train else 1.0
        
        print(len(pairs_list))
        self.transformer_ae=transforms.Compose(
            [
            transforms.RandomResizedCrop(
                (256,256),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.cond_transform = transforms.Compose([
            
            transforms.RandomResizedCrop(
                (256,256),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.ref_transform_people = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                [0.26862954, 0.26130258, 0.27577711]),
        ])


        self.ref_transform_back = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                [0.26862954, 0.26130258, 0.27577711]),
        ])

        # self.transformer_clip=CLIPImageProcessor.from_pretrained('/root/data1/github/pbp/sd-image-variations-diffusers/feature_extract')



    def __len__(self,):
        return len(self.data_pairs)

    def augmentation(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)   #确保每次产生的随机数是相同的
        if  transform1 is not None:
            frame_transform1 = transform1(frame) 
        else: 
            frame_transform1=frame
        
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)

    def __getitem__(self, index):
        back,local,people,pose,raw=self.data_pairs[index]
        back=Image.open(back)
        pose=Image.open(pose)
        raw=Image.open(raw)
        people=Image.open(people)
        
        # print(raw.size)
        if raw.size[0]>raw.size[1]:  # w>h
            transform1=self.random_square_width
        elif raw.size[0]<raw.size[1]: #h>w:
            transform1=self.random_square_height
        else:
            transform1=None
        
        state = torch.get_rng_state()
        

        raw = self.augmentation(raw, transform1, self.transformer_ae, state)
        pose=  self.augmentation(pose, transform1, self.cond_transform, state)
        people=self.augmentation(people,transform1,self.ref_transform_people,state)
        back=self.augmentation(back,transform1,self.ref_transform_back,state)
        ref_local_img=[]
        for i in os.listdir(local):
            ref_local_img.append(self.augmentation(Image.open(os.path.join(local,i)),transform1,self.transformer_ae,state))

        ref_local_img=torch.cat(ref_local_img,dim=0)


        return back,ref_local_img,people,pose,raw