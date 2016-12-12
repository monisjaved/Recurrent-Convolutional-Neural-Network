# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:25:41 2015

@author: Moonis Javed
"""


import theano

from support_math import *
from tweet_preprocess import TweetPreProcess
from rcnn_class import RCNN


t = TweetPreProcess()
x, y = t.create_set(filename="data/1700.csv", random=True)

vector_dict = theano.shared(np.asarray(t.word_vec).astype("float32"), name="vector_dict")

train_x_set = [x[i][:] for i in range(1000)]
train_x = [y[i][:] for i in range(1000)]

valid_x_set = [x[i][:] for i in range(1000, len(x))]
valid_x = [y[i][:] for i in range(1000, len(x))]

model = RCNN(vector_dict)
model.process_set(train_x_set,
                  train_x,
                  num_epochs=64,
                  verbose=True,
                  verbose_interval=1,
                  file_name="model_ada_small_check3",
                  valid_x=valid_x_set,
                  valid_y=valid_x)
