# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:32:55 2020

@author: Administrator
"""

import pickle as pk
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


with open('embedding.pkl', 'rb') as f:
    embedding_dict = pk.load(f)

a = []
mid = []
index = 0
for key,value in embedding_dict.items():
    a.append(value)
    mid.append(key)
    index+=1
mid = np.array(mid)
res = np.array(cosine_distances(a))

res = np.argsort(res, axis=1)
for i in range(200):
    print(mid[i])
    for j in range(10):
        print(mid[res[i][j]],end='--')
        
    print()
    print('='*20)

np.set_printoptions(suppress=True)
si_info = ['node_num','age','edu','sal','height','mar']
for i in range(1,len(si_info)):
    with open(si_info[i]+'_emb.pkl','rb') as f:
        embedding = pk.load(f)
    #print(np.array(embedding.numpy()))
    res = np.around(np.array(cosine_distances(embedding.numpy())),10)
    print(res)
