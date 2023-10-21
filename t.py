import os

with open('/root/data1/github/pbp/train_pairs.txt','r') as f:
    img=f.readlines()

with open('/root/data1/github/pbp/train_pairs_1.txt','w') as f:
    for i in img:
        l=list(i)
        l[-4:]='png\n'
        i=''.join(l)
        f.write(i)
