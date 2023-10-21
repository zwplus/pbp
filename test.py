# from unet_2d_condition import UNet2DConditionModel as unet
# from control_net import ControlNetModel
# # net=unet.from_pretrained('/home/user/zwplus/paper/dual/raw')
# net=unet.from_config('/root/data1/github/pbp/sd-image-variations-diffusers/dual_unet/config.json')
# # print(net.config.in_channels)
# # pose_control=ControlNetModel.from_unet(net)
# # print(pose_control.config)
# for name,param in net.named_parameters():
#     print(name)

import pytorch_lightning as pl
import torch
from torch import nn
from collections import OrderedDict


ck=torch.load('/root/data1/github/pbp/sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin')
print(ck['mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight'])

# print(ck.keys())
# print(ck['mid_block.back_attentions.0.proj_out.weight'])
# print(ck['mid_block.attentions.0.proj_out.weight'])
# target_dict=OrderedDict()
# back_dict=OrderedDict()
# for i in ck.keys():
#     if 'transformer_blocks' in str(i) and 'attn2' in str(i):
#         target_dict[i]=ck[i]
#         n=i.split('.')
#         name=[]
#         for j in n:
#             if j!='attn2':
#                 name.append(j)
#             else:
#                 name.append('attn3')
#         print('.'.join(name))
#         back_dict['.'.join(name)]=ck[i]
#     else:
#         if len(back_dict)>0:
#             for name in back_dict.keys():
#                 target_dict[name]=back_dict[name]
#             back_dict=OrderedDict()
#         target_dict[i]=ck[i]

# torch.save(target_dict,'/root/data1/github/pbp/sd-image-variations-diffusers/dual_unet/diffusion_pytorch_model.bin')

