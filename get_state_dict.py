import pytorch_lightning as pl
import torch
from torch import nn

ck=torch.load('/root/data1/github/pbp/sd-image-variations-diffusers/image_encoder/pytorch_model.bin')
print(ck.keys())
# print(ck['conv.weight'])
visual_projection={
    'weight':ck['visual_projection.weight'],
}
# feature_fusing={
#     'conv.weight':ck['conv.weight'],
#     'conv.bias':ck['conv.bias'],
#     'norm.weight':ck['norm.weight'],
#     'norm.bias':ck['norm.bias'],
#     'attn.to_q.weight':ck['attn.to_q.weight'],
#     'attn.to_k.weight':ck['attn.to_k.weight'],
#     'attn.to_v.weight':ck['attn.to_v.weight'],
#     'attn.to_out.0.weight':ck['attn.to_out.0.weight']
# }

torch.save(visual_projection,'/root/data1/github/pbp/sd-image-variations-diffusers/raw_proj/pro_j.ckpt')
# torch.save(feature_fusing,'/home/user/zwplus/paper/weight/people_feature/people_feature.ckpt')

# class CLIP_Proj(nn.Module):
#     def __init__(self,in_channel:int,out_channel:int,ck_path) -> None:
#         super().__init__()
#         self.refer_proj=nn.Linear(in_channel,out_channel,bias=False)
#         self.refer_proj.load_state_dict(torch.load(ck_path))
    
#     def forward(self,last_hidden_states_norm,num_images_per_prompt=1):
#         image_embeddings = self.refer_clip_proj(last_hidden_states_norm)
#         # duplicate image embeddings for each generation per prompt, using mps friendly method
#         bs_embed, seq_len, _ = image_embeddings.shape
#         image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
#         image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
#         return image_embeddings    



# # ck=torch.load('/home/user/zwplus/paper/weight/raw_pro_j/pro_j.ckpt')
# # print(ck['weight'].shape)
# pro_j=CLIP_Proj(1024,768,'/home/user/zwplus/paper/weight/raw_pro_j/pro_j.ckpt')
# print(pro_j.refer_proj.state_dict())






# ck=torch.load('/home/user/zwplus/paper/checkpoint/ddpm-epoch=009-fid=24.360.ckpt')
# style_encoder_dict={}
# print(ck['state_dict'].keys())
# # for i in ck['state_dict'].keys():
# #     if 'style_encoder' == i.split('.')[0]:
# #         style_encoder_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]
# fusion_dict={}
# for i in ck['state_dict'].keys():
#     if 'fusion' == i.split('.')[0]:
#         fusion_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]     
        
# model_dict={}
# for i in ck['state_dict'].keys():
#     if 'model' == i.split('.')[0]:
#         model_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]

# clip_encoder_dict={}
# for i in ck['state_dict'].keys():
#     if 'clip_encoder' == i.split('.')[0]:
#         clip_encoder_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]

# background_encoder_dict={}
# for i in ck['state_dict'].keys():
#     if 'background_encoder' == i.split('.')[0]:
#         background_encoder_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]

# person_encoder_dict={}
# for i in ck['state_dict'].keys():
#     if 'person_encoder' == i.split('.')[0]:
#         person_encoder_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]


# torch.save(clip_encoder_dict,'/home/user/zwplus/paper/checkpoint/clip_encoder_2.ck')   
# torch.save(fusion_dict,'/home/user/zwplus/paper/checkpoint/fusion_2.ck')
# torch.save(model_dict,'/home/user/zwplus/paper/checkpoint/unet_2.ck')
# torch.save(person_encoder_dict,'/home/user/zwplus/paper/checkpoint/person_encoder_2.ck')
# torch.save(background_encoder_dict,'/home/user/zwplus/paper/checkpoint/background_encoder_2.ck')