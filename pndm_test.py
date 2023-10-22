import sys
import os
from typing import Any, Dict,List,Optional
import random
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.utils import make_grid
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import AutoencoderKL,PNDMScheduler,DDIMScheduler,DDPMScheduler
from pytorch_lightning import seed_everything
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from pytorch_lightning.strategies.ddp import DDPStrategy

from unet_2d_condition import UNet2DConditionModel as unet
from style_encoder import CLIP_Image_Extractor,CLIP_Proj,people_global_fusion,people_local_fusion
from diffusers.utils.import_utils import is_xformers_available
from tiktok_dataset_2 import diffusion_dataset
from control_net import ControlNetModel

seed_everything(24)

class People_Background(pl.LightningModule):
    def __init__(self,
                unet_config:Dict=None,
                people_config:Dict=None,
                background_config:Optional[Dict]=None,
                vae_path:str=None,
                train_stage:str='people',out_path='',
                image_size=(4,32,32),condition_rate=0.1,condition_guidance=3,
                warm_up=6000,learning_rate=1e-4,local_num=8,enable_xformers_memory_efficient_attention=True):
        super().__init__()
        self.train_stage=train_stage
        torch.set_float32_matmul_precision('high')
        self.init_model(unet_config,people_config,background_config,vae_path,train_stage,enable_xformers_memory_efficient_attention)
        self.scheduler=PNDMScheduler()
        self.scheduler.set_timesteps(50)

        self.condition_rate=condition_rate
        self.condition_guidance=condition_guidance
        self.warm_up=warm_up
        self.lr=learning_rate
        self.local_num=local_num
        

        self.out_path=out_path
        self.laten_shape=image_size
        self.save_img_num=0
        self.tr=transforms.ToPILImage()

    def init_model(self,unet_config,people_config,background_config:Optional[Dict]=None,vae_path:str=None,train_stage:str=None,
                enable_xformers_memory_efficient_attention:bool=True,gradient_checkpointing=True):
        self.laten_model=AutoencoderKL.from_pretrained(vae_path)
        self.laten_model.eval()
        self.laten_model.requires_grad_(False)

        self.unet=unet.from_pretrained(unet_config['ck_path'])

        self.clip=CLIP_Image_Extractor(**people_config['clip_image_extractor'])
        self.clip.eval()
        self.clip.requires_grad_(False)
        
        self.people_proj=CLIP_Proj(**people_config['clip_proj'])
        self.people_global_fusion=people_global_fusion(**people_config['global_fusion'])
        self.people_local_fusion=people_local_fusion(**people_config['local_fusion'])

        
        #初始化background_config
        # if background_config is not None:
        self.back_proj=CLIP_Proj(**background_config['clip_proj'])

        self.controlnet_pose = ControlNetModel.from_unet(unet=self.unet)
        # if gradient_checkpointing:
        #     self.controlnet_pose.enable_gradient_checkpointing()

        need_train=0
        # if train_stage=='people':
        #     for name,param in self.unet.named_parameters():
        #         if 'attentions'  in name and 'transformer_blocks'  in name:
        #             param.requires_grad_(True)
        #             need_train+=1
        #         elif 'transformer_blocks' not in name:
        #             param.requires_grad_(False)
        #         else:
        #             param.requires_grad_(False)
        # elif train_stage=='back':
        #     self.people_proj.eval()
        #     self.people_proj.requires_grad_(False)
        #     self.people_global_fusion.eval()
        #     self.people_global_fusion.requires_grad_(False)
        #     self.people_local_fusion.eval()
        #     self.people_local_fusion.requires_grad_(False)

        #     for name,param in self.unet.named_parameters():
        #         if 'back_attentions'  in name and 'transformer_blocks'  in name:
        #             param.requires_grad_(True)
        #             need_train+=1
        #         else:
        #             param.requires_grad_(False)
        # elif train_stage=='people_back':
        #     for name,param in self.unet.named_parameters():
        #         if 'transformer_blocks'  in name:
        #             param.requires_grad_(True)
        #             need_train+=1
        #         else:
        #             param.requires_grad_(False)
        print(f'There are {need_train} modules in unet to be set as requires_grad=True.')

        self.fid=FrechetInceptionDistance(normalize=True)
        self.lpips=LPIPS(net_type='vgg',normalize=True)
        self.ssim=SSIM(data_range=1.0)
        self.psnr=PSNR(data_range=1.0)

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                print('start xformer')
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet_pose.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")

    def training_step(self, batch, batch_idx):
        self.save_img_num=0
        rate=random.random()


        background_img,part_img,people_img,pose_img,img=batch
        img=img.to(self.device)
        people_img=people_img.to(self.device)
        part_img=part_img.to(self.device)
        background_img=background_img.to(self.device)
        pose_img=pose_img.to(self.device)
            

        if rate <= self.condition_rate:
            people_feature=self.get_people_condition(
                torch.zeros_like(people_img).to(self.device),
                torch.zeros_like(part_img).to(self.device))
            back_feature=self.get_back_feature(torch.zeros_like(background_img).to(self.device))
        else:
            people_feature=self.get_people_condition(people_img,part_img)
            back_feature=self.get_back_feature(background_img)
        
        target=self.img_to_laten(img).detach()
        noise=torch.randn(target.shape).to(self.device)
        timesteps=torch.randint(0,self.scheduler.config.num_train_timesteps,(target.shape[0],)).long().to(self.device)
        noisy_image=self.scheduler.add_noise(target,noise,timesteps)
        

        model_out = self(noisy_image, timesteps,people_feature,back_feature,pose_img)

        loss=F.mse_loss(model_out,noise)

        self.log('train_loss',loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def forward(self,laten:torch.FloatTensor=None,timesteps:torch.Tensor=None,
                people_feature:torch.FloatTensor=None,back_feature:Optional[torch.FloatTensor]=None,pose_img:torch.FloatTensor=None): 
                down_block_res_samples, mid_block_res_sample=self.controlnet_pose(
                        sample=laten,timestep=timesteps,
                        encoder_hidden_states=people_feature, # both controlnet path use the refer latents
                        controlnet_cond=pose_img, conditioning_scale=1.0, return_dict=False,back_hidden_states=back_feature)
                return self.unet(sample=laten,timestep=timesteps,
                        encoder_hidden_states=people_feature,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        back_hidden_states=back_feature).sample
    
    @torch.no_grad()
    def sample(self,people_img:torch.FloatTensor,part_img:torch.FloatTensor,
                background_img:Optional[torch.FloatTensor]=None,pose_img:torch.FloatTensor=None):

            latens_=torch.randn([people_img.shape[0],*self.laten_shape]).to(self.device)

            uncond_people_feature=self.get_people_condition(torch.zeros_like(people_img).to(self.device),
                                                            torch.zeros_like(part_img).to(self.device))
            cond_people_feature=self.get_people_condition(people_img,part_img)
            people_feature=torch.cat([cond_people_feature,uncond_people_feature])

            uncond_back_feature=self.get_back_feature(torch.zeros_like(background_img).to(self.device))
            cond_back_feature=self.get_back_feature(background_img).detach()
            back_feature=torch.cat([cond_back_feature,uncond_back_feature])
            
            pose_img=torch.cat([pose_img,pose_img])

            for t in tqdm(self.scheduler.timesteps):
                latens=torch.cat([latens_]*2)
                timestep=torch.full((latens.shape[0],),t).to(self.device)
                noise_pred=self(latens,timestep,people_feature,back_feature,pose_img)
                noise_cond,noise_uncond=noise_pred.chunk(2)
                noise_pred=noise_uncond+self.condition_guidance*(noise_cond-noise_uncond)
                latens_=self.scheduler.step(noise_pred,t,latens_).prev_sample
            return latens_

    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        rate=random.random()

        background_img,part_img,people_img,pose_img,img=batch
        img=img.to(self.device)
        people_img=people_img.to(self.device)
        part_img=part_img.to(self.device)
        background_img=background_img.to(self.device)
        pose_img=pose_img.to(self.device)
        
        if rate<1:
            target_img=self.sample(people_img,part_img,background_img,pose_img)
            target_img=self.laten_to_img(target_img)
            target_img=torch.clamp(target_img.detach()/2+0.5,0,1).detach()
            img=img.detach()/2+0.5
            pose_img=pose_img.detach().cpu()/2+0.5
            
            
            self.fid.update(target_img,real=False)
            self.fid.update(img,real=True)
            self.log('fid',self.fid,prog_bar=True,logger=True,on_step=True,on_epoch=True)

            self.lpips.update(target_img,img)
            self.log('lpips',self.lpips,prog_bar=True,logger=True,on_step=True,on_epoch=True)
            self.psnr.update(target_img,img)
            self.log('psnr',self.psnr,prog_bar=True,logger=True,on_step=True,on_epoch=True)
            self.ssim.update(target_img,img)
            self.log('ssim',self.ssim,prog_bar=True,logger=True,on_step=True,on_epoch=True)

            img=img.detach().cpu()
            pose_img=pose_img.detach().cpu()
            target_img=target_img.detach().cpu()
            # part_img=part_img.detach().cpu()



            file_dir=os.path.join(self.out_path,str(self.global_step))
            os.makedirs(file_dir,exist_ok=True)
            for i in range(target_img.shape[0]):
                save_img=self.tr(target_img[i])
                save_img.save(os.path.join(file_dir,str(self.save_img_num)+'.jpg'))
                self.save_img_num+=1
                if self.save_img_num==1:
                    # n_img=part_img[:4]/2+0.5
                    # n_img=rearrange(n_img,'b (l c) h w -> (b l) c h w',l=8).contiguous()
                    # n_img=torch.stack([torch.sum(n_img[i:i+8],dim=0) for i in range(0,n_img.shape[0],8)])

                    h=torch.cat([img[:4],pose_img[:4],target_img[:4]])
                    show_img=make_grid(h,nrow=4,padding=1)
                    show_img=torch.unsqueeze(show_img,dim=0)
                    logger.experiment.add_images('val/img',show_img,self.global_step)
    @torch.no_grad()
    def test_step(self,batch,batch_idx):
        rate=random.random()

        background_img,part_img,people_img,pose_img,img=batch
        img=img.to(self.device)
        people_img=people_img.to(self.device)
        part_img=part_img.to(self.device)
        background_img=background_img.to(self.device)
        pose_img=pose_img.to(self.device)
        
        if rate<1:
            target_img=self.sample(people_img,part_img,background_img,pose_img)
            target_img=self.laten_to_img(target_img)
            target_img=torch.clamp(target_img.detach()/2+0.5,0,1).detach()
            img=img.detach()/2+0.5
            pose_img=pose_img.detach().cpu()/2+0.5
            
            
            self.fid.update(target_img,real=False)
            self.fid.update(img,real=True)
            self.log('fid',self.fid,prog_bar=True,logger=True,on_step=True,on_epoch=True)

            self.lpips.update(target_img,img)
            self.log('lpips',self.lpips,prog_bar=True,logger=True,on_step=True,on_epoch=True)
            self.psnr.update(target_img,img)
            self.log('psnr',self.psnr,prog_bar=True,logger=True,on_step=True,on_epoch=True)
            self.ssim.update(target_img,img)
            self.log('ssim',self.ssim,prog_bar=True,logger=True,on_step=True,on_epoch=True)

            img=img.detach().cpu()
            pose_img=pose_img.detach().cpu()
            target_img=target_img.detach().cpu()
            # part_img=part_img.detach().cpu()



            file_dir=os.path.join(self.out_path,str(self.global_step))
            os.makedirs(file_dir,exist_ok=True)
            for i in range(target_img.shape[0]):
                save_img=self.tr(target_img[i])
                save_img.save(os.path.join(file_dir,str(self.save_img_num)+'.jpg'))
                self.save_img_num+=1
                if self.save_img_num==1:
                    # n_img=part_img[:4]/2+0.5
                    # n_img=rearrange(n_img,'b (l c) h w -> (b l) c h w',l=8).contiguous()
                    # n_img=torch.stack([torch.sum(n_img[i:i+8],dim=0) for i in range(0,n_img.shape[0],8)])

                    h=torch.cat([img[:4],pose_img[:4],target_img[:4]])
                    show_img=make_grid(h,nrow=4,padding=1)
                    show_img=torch.unsqueeze(show_img,dim=0)
                    logger.experiment.add_images('val/img',show_img,self.global_step)


    def get_people_condition(self,people_img,part_img):
        part_img=rearrange(part_img,'b (l c) h w -> (b l) c h w',c=3).contiguous()
        part_laten=self.laten_model.encoder.get_feature(part_img).detach()

        people_global_feature=self.clip(people_img).detach()
        people_global_feature=self.people_proj(people_global_feature)

        people_global_feature=self.people_global_fusion(part_laten,people_global_feature)
        people_local_feature=self.people_local_fusion(people_global_feature)

        return people_local_feature

    def get_back_feature(self,background_img):
        back_feature=self.clip(background_img).detach()
        back_feature=self.back_proj(back_feature)
        return back_feature
    
    def configure_optimizers(self):

        params =[i  for i in (list(self.people_proj.parameters())+list(self.people_global_fusion.parameters())
                +list(self.people_local_fusion.parameters())+list(self.unet.parameters())+list(self.back_proj.parameters())
                +list(self.controlnet_pose.parameters()))
                    if i.requires_grad==True ]
        optim = torch.optim.AdamW(params, lr=self.lr)
        lambda_lr=lambda step: max((self.global_step)/self.warm_up,1e-4) if (self.global_step)< self.warm_up else max((63000-self.global_step)/(63000-self.warm_up),1e-3)
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda_lr)
        return {'optimizer':optim,'lr_scheduler':{"scheduler":lr_scheduler,'monitor':'fid','interval':'step','frequency':1}}

    @torch.no_grad()
    def img_to_laten(self,imgs):
        latens=self.laten_model.encode(imgs).latent_dist.sample().detach()
        latens=0.18215*latens
        return latens
    
    @torch.no_grad()
    def laten_to_img(self,latens):
        latens=1/0.18215*latens
        return self.laten_model.decode(latens).sample.detach()


train_list=[
    '/root/data1/github/pbp/train_pairs_1.txt'
]
test_list=[
    '/root/data1/github/pbp/test_pairs_1.txt'
]



if __name__=='__main__':
    logger=TensorBoardLogger(save_dir='/root/data1/github/pbp/log/')
    train_dataset=diffusion_dataset(train_list,if_train=True)
    test_dataset=diffusion_dataset(test_list,if_train=True)

    train_loader=DataLoader(train_dataset,batch_size=6,shuffle=True,pin_memory=True,num_workers=8)
    val_loader=DataLoader(test_dataset,batch_size=6,pin_memory=True,num_workers=8,drop_last=True)
    
    unet_config={
        'ck_path':'/root/data1/github/pbp/sd-image-variations-diffusers/dual_unet/',
    }
    people_config={
        'clip_image_extractor':
            {
                'clip_path':'/root/data1/github/pbp/sd-image-variations-diffusers/image_encoder'
            },
        'clip_proj':{
            'in_channel':1024,
            'out_channel':768,
            'ck_path':'/root/data1/github/pbp/sd-image-variations-diffusers/raw_proj/pro_j.ckpt'
        },
        'global_fusion':{
            'inchannels':512,
            'ch':768,
            'local_num':8,
            'heads':8,
        },
        'local_fusion':{
            'inchannels':768,
            'mult':2,
            'local_num':8,
            'heads':8,
            'head_dim':128,
        }
    }

    background_config={
        'clip_proj':{
            'in_channel':1024,
            'out_channel':768,
            'ck_path':'/root/data1/github/pbp/sd-image-variations-diffusers/raw_proj/pro_j.ckpt'
        }
    }

    vae_path='/root/data1/github/pbp/sd-image-variations-diffusers/vae_2/'
    logger=TensorBoardLogger(save_dir='/root/data1/github/pbp/')

    model=People_Background(unet_config,people_config,background_config,vae_path=vae_path,
                            train_stage='people',out_path='/root/data1/github/pbp/ouput')


    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="/root/data1/github/pbp/checkpoint", save_top_k=5, monitor="fid",mode='min',filename="pndm-{epoch:03d}-{fid:.3f}")
    
    trainer=pl.Trainer(
        logger=logger,callbacks=[checkpoint_callback],default_root_dir='/root/data1/github/pbp/checkpoint',
        strategy='ddp_find_unused_parameters_true',accelerator='gpu',devices=4,accumulate_grad_batches=8,check_val_every_n_epoch=4,
        log_every_n_steps=1000,max_epochs=200,
        profiler='simple',benchmark=True,gradient_clip_val=1) 
    
    trainer.fit(model,train_loader,val_loader,ckpt_path='/root/data1/github/pbp/checkpoint/pndm-epoch=075-fid=65.417.ckpt') 