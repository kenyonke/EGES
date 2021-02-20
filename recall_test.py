# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:15:54 2020

@author: Administrator
"""
import random
import pickle as pk
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def coldStart(age_e, height_e, sal_e, edu_e, mar_e, age, edu, sal, height, mar, is_processed=True):
    
    if not is_processed:
        one_hot_fea = []
        
        # age [0-7]
        if age == '0' or age == '-1':
            one_hot_fea.append(0)
        elif age <= '25':
            one_hot_fea.append(1)
        elif age <= '30':
            one_hot_fea.append(2)
        elif age <= '35':
            one_hot_fea.append(3)
        elif age <= '40':
            one_hot_fea.append(4)
        elif age <= '50':
            one_hot_fea.append(5)
        elif age <= '60':
            one_hot_fea.append(6)
        else:
            one_hot_fea.append(7)
            
        # edu [0, 2-7]
        one_hot_fea.append(0 if edu=='0' or edu=='-1' else int(edu)-1)
        
        # sal [0, 3-9]
        one_hot_fea.append(0 if sal=='0' or sal=='-1' else int(sal)-2)
        
        # height [0-11]
        if height == '0' or height =='-1':
            one_hot_fea.append(0)
        elif height <= '150':
            one_hot_fea.append(1)
        elif height <= '155':
            one_hot_fea.append(2)
        elif height <= '160':
            one_hot_fea.append(3)
        elif height <= '165':
            one_hot_fea.append(4)
        elif height <= '170':
            one_hot_fea.append(5)
        elif height <= '170':
            one_hot_fea.append(6)
        elif height <= '175':
            one_hot_fea.append(7)
        elif height <= '180':
            one_hot_fea.append(8)  
        elif height <= '185':
            one_hot_fea.append(9)  
        elif height <= '190':
            one_hot_fea.append(10)  
        else:
            one_hot_fea.append(11)
        
        # mar [0, 1 ,3 ,4]
        one_hot_fea.append(int(mar) if mar in ('0','1') else int(mar)-1)
        
    else:
        one_hot_fea = [int(age),  int(edu), int(sal), int(height), int(mar)]
        
    # average pooling
    #print(one_hot_fea)
    embedding = age_e[one_hot_fea[0]]
    embedding += edu_e[one_hot_fea[1]]
    embedding += sal_e[one_hot_fea[2]]
    embedding += height_e[one_hot_fea[3]]
    embedding += mar_e[one_hot_fea[4]]
    embedding /= 5
    
    return np.array(embedding)

file = 'women/'

#主动方average pooling embedding所需数据
mids = {}
with open('sz_women.txt','r') as f:
    for line in f.readlines():
        mid,oid = line.split()[:2]
        if mid not in mids:
            mids[mid] = [oid]
        else:
            mids[mid].append(oid)

'''
# 取5000主动方id去检验效果
mid_list = list(mids.keys())
random.shuffle(mid_list)
recall_mid = mid_list[:5000]
del mid_list
with open('women_recall_list.txt', 'w') as file:
    for mid in recall_mid:
        file.write(str(mid) + '\n')
'''


# 冷启动需要用到的features
with open( file+ 'info.pkl','rb') as f:
    info_dict = pk.load(f)

# 无需冷启动的embedding
with open( file+ 'embedding.pkl', 'rb') as f:
    embedding_dict = pk.load(f)

# side information embedding

with open( file+ 'age_emb.pkl','rb') as f:
    age_e = pk.load(f)
with open( file+ 'height_emb.pkl','rb') as f:
    height_e = pk.load(f)
with open( file+ 'edu_emb.pkl','rb') as f:
    edu_e = pk.load(f)
with open( file+ 'sal_emb.pkl','rb') as f:
    sal_e = pk.load(f)
with open( file+ 'mar_emb.pkl','rb') as f:
    mar_e = pk.load(f)


#print(coldStart(age_e, height_e, sal_e, edu_e, mar_e, 1,1,1,1,1))

# 5000个主动方
memberid_list = []
with open('women_recall_list.txt', 'r') as f:
    for line in f.readlines():
        memberid_list.append(line.strip())


# 召回
recall_list = np.load('np_women_recall.npy')

with open('eges_res.txt', 'w') as file:
    for i in range(len(memberid_list)):
        # 主动方average pooling
        mid = memberid_list[i]
        emd = np.zeros(len(age_e[0]))
        for oid in mids[mid][:10]:
            if oid in embedding_dict:
                emd += embedding_dict[oid]
            else:
                o_info = info_dict[oid]
                emd += coldStart(age_e, height_e, sal_e, edu_e, mar_e, o_info[0], o_info[1], o_info[2], o_info[3], o_info[4])
        emd /= len(mids[mid][:10])
        
        # 召回
        recall = recall_list[i]
        recall_emb = []
        for o_info in recall:
            if o_info[0] in embedding_dict:
                recall_emb.append(embedding_dict[o_info[0]])
            else:
                #print(o_info[1], o_info[2], o_info[3], o_info[4], o_info[5])
                age, height, sal, edu, mar = o_info[1:6]
                recall_emb.append(coldStart(age_e, height_e, sal_e, edu_e, mar_e, age, edu, sal, height, mar, is_processed=False))
        
        # 统计cosine_distances计算粗排结果
        out = np.argsort(cosine_distances(recall_emb+[emd])[:-1,-1])
        res = []
        for i,score in enumerate(out):
            if score <= 1000:
                res.append(recall[i][0])
        
        for oid in res:
            file.write(mid+'\t'+oid+'\n')
