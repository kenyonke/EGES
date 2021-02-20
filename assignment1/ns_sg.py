# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:59:31 2020

@author: Administrator
"""



import numpy as np
from math import exp

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    ng_indices = [target]
    ng_indices.extend(getNegativeSamples(target, dataset, K))
    del(ng_indices[0])
    grad = np.zeros(outputVectors.shape,dtype=np.float)
    
    prob = sigmoid([np.matmul(outputVectors[target],predicted.T)])[0]
    gradPred = (prob-1) * outputVectors[target]
    grad[target] = (prob-1) * predicted
    cost = -np.log(prob)
    
    for ng_index in ng_indices:
        prob = sigmoid([np.matmul(outputVectors[ng_index],predicted.T)])[0]
        cost -= np.log(1 - prob)
        gradPred += prob * outputVectors[ng_index]
        grad[ng_index] += prob * predicted

    return cost, gradPred, grad

def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def sigmoid_grad(s):
    s = np.array(s, dtype=np.float)
    ds = s * (1 - s)
    return ds


def sgd(f, x0, learning_rate, epochs=200, postprocessing=lambda x:x, PRINT_EVERY=10):
    ANNEAL_EVERY = 20000
    x = x0
    expcost = None

    for i in range(epochs):

        cost,grad = f(x)
        x -= grad * learning_rate
        postprocessing(x)
        
        if i!=0 and i % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d loss: %f" % (i, expcost))
        
        # 模拟退火
        if i!=0 and i % ANNEAL_EVERY == 0:
            learning_rate *= 0.5

    return x


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=negSamplingCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    cur_id = tokens[currentWord]
    V_c = inputVectors[cur_id]
    
    #window for context words
    for contextWord in contextWords:
        target = tokens[contextWord]
        WordCost,gradPred,grad = word2vecCostAndGradient(V_c, target, outputVectors, dataset)
        cost += WordCost
        gradIn[cur_id] += gradPred
        gradOut += grad

    return cost, gradIn, gradOut

def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = np.array(x,dtype=np.float)
        c = -np.max(x,axis=1)
        for i in range(x.shape[0]):
            x[i] = np.exp(x[i]+c[i])/np.sum(np.exp(x[i]+c[i]))
    else:
        # Vector
        x = np.array(x,dtype=np.float)
        c = -x[np.argmax(x)]
        x = np.exp(x+c)/np.sum(np.exp(x+c))
        
    assert x.shape == orig_shape
    return x

def sigmoid(x):
    x = np.array(x,dtype=np.float)
    s = np.zeros(x.shape)
    if(len(x.shape))==1:
        #vector
        for i in range(x.shape[0]):
            s[i] = 1/(1+exp(-x[i]))
        
    else:
        #matrix
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i,j] = 1/(1+exp(-x[i,j]))
    return s

def forward_backward_prop(X, labels, params, dimensions):
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
    #forward propagation
    Z1 = np.matmul(X,W1) + b1
    a1 = sigmoid(Z1)    
    Z2 = np.matmul(a1,W2) + b2
    a2 = softmax(Z2)

    cost = -1/labels.shape[0] * np.sum(np.sum(np.log(a2) * labels))
    print('cost: ',cost)
    
    
    #backward propagation
    
    #matrix with the whole data
    e2 = (a2-labels)/labels.shape[0]
    gradb2 = np.sum(e2,axis=0)
    gradW2 = np.matmul(a1.T ,e2)
    
    e1 = np.matmul(e2,W2.T) * sigmoid_grad(a1)
    gradb1 = np.sum(e1,axis=0)
    gradW1 = np.matmul(X.T, e1)
    
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad