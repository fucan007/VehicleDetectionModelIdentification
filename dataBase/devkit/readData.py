#-*- coding:utf-8 -*-
import sys
import pickle
import struct

import scipy.io

data = scipy.io.loadmat('./cars_train_annos.mat')  # 读取mat文件

#dic = {}
dic = data['annotations'].tolist()
test = data['annotations'].item(5)
print (test)
filename = test[5]
print (filename)
#print (dic[0])
#print (len(data['annotations']['class']))
for k,v  in data.items():
    print (k)
print (len(data.items()))
#if data['annotations']['fname'].any() == '00001.jpg':