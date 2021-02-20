# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:05:22 2020

@author: Administrator
"""

import random
import networkx as nx
from collections import Counter

# 实现walk功能的一个Graph类
class Graph:
    
    def __init__(self, data):
        self.G = self.load_data(data)
        self.mid_freq = self.freq(data)
        self.get_graph_info()
          
        
    def load_data(self,data):
        G = nx.DiGraph()
        for index in range(len(data)):
            for i in range(len(data[index][1])-1):
                l_node = data[index][1][i]
                r_node = data[index][1][i+1]
                if (l_node, r_node) in G.edges():
                    G.add_edge(l_node, r_node, weight=G[l_node][r_node]['weight']+1)
                else:
                    G.add_edge(l_node, r_node, weight=1)
        return G
        
    
    def freq(self,data):
        mid_freq = Counter()
        for i in range(len(data)):
            mid_freq.update(data[i][1])
        return mid_freq
    
    
    def get_graph_info(self):
        #nx.draw(G,with_labels=True)
        print('number of edges:', len(self.G.edges()))
        s = 0
        for u,v,d in self.G.edges(data = 'weight'):
            if d>1:
                s+=1
        print('number of edges with weight greater than 1:',s)
        print('number of edges:', len(self.G.edges()))
        print('number of nodes:', len(self.G.nodes()))
        #print(nx.adjacency_matrix(self.G))
        
        
    # weighted deep walk
    def random_walk(self, path_length, start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            start: the start node of the random walk.
        """
        if start:
            path = [start]
        else:
            # starts from a random node if start is not defined
            path = [random.choice(list(self.G.nodes()))]
        
        while len(path) < path_length:
            neighbor_dict = self.G[str(path[-1])]
            if len(neighbor_dict.keys())!=0: # 有临近点的时候开始walk
                walk_list = []   # 将所有临近点根据weight全部保存进一个list，然后随机选择node作为一个walk
                for node,weight in neighbor_dict.items():
                    for i in range(weight['weight']):
                        walk_list.append(node)
                path.append(random.choice(walk_list)) #权重walk
            else:  #无临近点直接退出
                break 
        return path  
    
    
    # node2vec walk
    def node2vec_walk(self, path_length, start=None):
        pass

    
    # num_paths:每个node走多少次
    # path_length:每次walk的长度
    def build_deepwalk_corpus(self, num_paths, path_length):
        """ Returns a corpus(list) from deep walk.
            num_paths: number of path every node walks
            path_length: maximum length of a path 
        """
        walks = []
        nodes = list(self.G.nodes())
      
        for cnt in range(num_paths):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(path_length, start=node))
                
        return walks

    