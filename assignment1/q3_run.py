#!/usr/bin/env python
#-*- coding:utf-8 -*- 

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from q3_word2vec import word2vec_sgd_wrapper,skipgram,negSamplingCostAndGradient
from q3_sgd import sgd

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
print(tokens)