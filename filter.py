import random

with open('/root/data1/github/pbp/test_pairs.txt','r') as f:
    img_list=f.readlines()

e=open('/root/data1/github/pbp/new_test_pairs.txt','w')
for i in img_list:
    if random.random()>0.5:
        e.write(i)
