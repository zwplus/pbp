from unet_2d_condition import UNet2DConditionModel as unet
from control_net import ControlNetModel
import torch
from torch.optim import AdamW
net=unet.from_pretrained('/home/user/zwplus/pbp/sd-image-variations-diffusers/dtb_unet')
# opt=AdamW(list(net.parameters()),lr=0.0001)

# l=torch.randn((1,4,32,32))
# time_step=torch.ones((1,))
# e=torch.randn((1,1,768))
# b=torch.randn((1,256,768))
# n=torch.randn((1,4,32,32))

# opt.zero_grad()

# p=net(sample=l,timestep=time_step,
#     encoder_hidden_states=e,
#     back_hidden_states=b).sample
# loss=torch.nn.functional.l1_loss(p,n)
# loss.backward()
# for name, param in net.named_parameters():
#     if param.grad is None:
#         print(name)
# opt.step()

# net=unet.from_config('/home/user/zwplus/pbp/sd-image-variations-diffusers/dtb_unet/config.json')
# # print(net.config.in_channels)
# pose_control=ControlNetModel.from_unet(net)
# # # print(pose_control.config)
# for name,param in net.named_parameters():
#     print(name)

# import pytorch_lightning as pl
# import torch
# from torch import nn
# from collections import OrderedDict


# ck=torch.load('/home/user/zwplus/pbp/sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin')
# # print(ck['mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight'])

# # print(ck.keys())
# # print(ck['mid_block.back_attentions.0.proj_out.weight'])
# # print(ck['mid_block.attentions.0.proj_out.weight'])
# target_dict=OrderedDict()
# people_dict=OrderedDict()
# back_dict=OrderedDict()
# for i in ck.keys():
#     if 'transformer_blocks' in str(i) :
#         n=i.split('.')
#         name_people=[]
#         name_back=[]
#         for j in n:
#             if j!='transformer_blocks':
#                 name_back.append(j)
#                 name_people.append(j)
#             else:
#                 name_back.append('transformer_blocks_back')
#                 name_people.append('transformer_blocks_people')
#         print('.'.join(name_people))
#         back_dict['.'.join(name_back)]=ck[i]
#         people_dict['.'.join(name_people)]=ck[i]
#     else:
#         if len(people_dict)>0:
#             for name in people_dict.keys():
#                 target_dict[name]=people_dict[name]
#             people_dict=OrderedDict()
#         if len(back_dict)>0:
#             for name in back_dict.keys():
#                 target_dict[name]=back_dict[name]
#             back_dict=OrderedDict()
#         target_dict[i]=ck[i]

# torch.save(target_dict,'/home/user/zwplus/pbp/sd-image-variations-diffusers/dtb_unet/diffusion_pytorch_model.bin')

