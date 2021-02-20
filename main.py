import pickle as pk
import numpy as np
import time
from skip_gram import Eges_v2
from process_data import feature_process,process_data

if __name__ == '__main__':
    #训练参数
    epoch = 100     # 迭代次数
    embedding_size = 32                         # 特征embedding的size
    learning_rate =0.005                        # lr
    negative_sampling_num=5                     # 负采样次数
    feature_num = 6                             # 特征个数, 包括自身id作为特征
    feature_size_list = [8, 7, 8, 12, 4]        # 特征的one-hot size
    
    '''
    生成训练batch样本,储存为batch_x.pkl 和 batch_y.pkl两个文件
    同时也会生成
    所有用户的特征dict          文件名 info.pkl
    memberid转one-hot id的dict  文件名 mid_to_id.pkl
    one-hot id转memberid的dict  文件名 id_to_mid.pkl
    '''
    process_data(file_name='sz_men.txt',
                 batch_size=10000,
                 num_paths=2,
                 path_length=8,
                 window_size=4)
    
    with open('mid_to_id.pkl','rb') as f:
        mid_to_id = pk.load(f)
    feature_size_list = [len(mid_to_id.keys())] + feature_size_list  # 加上id 的one-hot
    
    s = time.time()
    # 构建模型
    model = Eges_v2(feature_num = feature_num,
                    feature_size_list = feature_size_list,  # [node_num age edu sal height mar]
                    learning_rate = learning_rate,
                    embedding_size = embedding_size)
    
    # 读取训练数据
    with open('batch_x.pkl', 'rb') as f:
        x = pk.load(f)
    with open('batch_y.pkl', 'rb') as f:
        y = pk.load(f)
    
    # 开始训练
    for ep in range(epoch):
        loss = 0
        for i in range(len(x)):
            loss += model.train(x[i], y[i], 1, negative_sampling_num=negative_sampling_num)
        print('epoch: '+ str(ep), ' loss: '+ str(loss))                                         
    
    
    # 提取embedding
    with open('info.pkl','rb') as f:
        info_dict = pk.load(f)
    with open('mid_to_id.pkl','rb') as f:
        mid_to_id = pk.load(f)
    with open('id_to_mid.pkl','rb') as f:
        id_to_mid = pk.load(f)
    
    usrs = []
    for key,val in info_dict.items():
        val = [str(v) for v in val]
        try:
            usrs.append([mid_to_id[key]] + feature_process(val))
        except:
            continue
    usrs = np.array(sorted(usrs, key=lambda x:x[0]))

    h = np.array(model.get_embedding(usrs))
    
    # 保存side information的embedding
    si_info = ['node_num','age','edu','sal','height','mar']
    for i in range(1,len(model.side_ws)):
        with open(si_info[i]+'_emb.pkl','wb') as f:
            pk.dump(model.side_ws[i], f)

    # 保存embedding
    embeddings = {}
    for i,h_i in enumerate(h):
        embeddings[id_to_mid[i]] = h_i
        
    with open('embedding.pkl','wb') as f:
        pk.dump(embeddings, f)
    print(time.time()-s)