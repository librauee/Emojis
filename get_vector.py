# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:33:27 2019

@author: Lee
"""

import tensorflow as tf
import pickle
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'
model_path='tensorflow_inception_graph.pb'
input_img_path='BQBCollection'
output_folder='data1/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def parse_img(input_img_path):
    """
    制作数据和标签
    
    """
    img_datas=[]
    img_labels=[]
    for i in os.listdir(input_img_path):
        if i[-3:]=='jpg':
            img_data=tf.gfile.FastGFile(os.path.join(input_img_path,i),'rb').read()
            img_datas.append(img_data)
            img_labels+=[i[:-4]]
    return np.array(img_datas),np.array(img_labels)

def load_inception_v3(model_path):
    """
    导入计算图  
    
    """
    with tf.gfile.FastGFile(model_path,'rb') as f:
        # 创建一张新图
        graph_def=tf.GraphDef()
        # 将打开的模型图写入到新图中
        graph_def.ParseFromString(f.read())
        # 将这张新图设为默认图
        _=tf.import_graph_def(graph_def=graph_def,name='')


batch_size=500
img_datas,img_labels=parse_img(input_img_path)
load_inception_v3(model_path)
num_batches=int(len(img_datas)/batch_size)

with tf.Session() as sess:
    """
    打开会话，进行特征提取
    
    """
    sess.run(tf.global_variables_initializer())
    # 通过名称获取张量
    tensor=sess.graph.get_tensor_by_name('pool_3/_reshape:0')
    # 每500张图片为一个批次循环处理
    for i in range(num_batches):
        batch_img_data=img_datas[i*batch_size:(i+1)*batch_size]
        batch_img_labels=img_labels[i*batch_size:(i+1)*batch_size]
        # 制作一个存放特征向量的空列表
        feature_v=[]
        # 将每一张图片转为的像素矩阵作为数据传入到tensor中做前向计算，得到2048的特征向量
        for j in batch_img_data:
            j_vector=sess.run(tensor,feed_dict={'DecodeJpeg/contents:0':j})
            # 逐个添加
            feature_v.append(j_vector)
        feature_v=np.vstack(feature_v)
        # 保存特征向量的全路径
        save_path=os.path.join(output_folder,'data_%d.pickle'%i)
        # 打开这个全路径文件
        with tf.gfile.FastGFile(save_path,'w') as f:
        	# 写入这个批次的向量，便于后续提取
            pickle.dump((feature_v,batch_img_labels),f)
        print(save_path,'is_ok!')
