# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:01:23 2019

@author: Lee
"""

from sklearn.cluster import KMeans
import numpy as np
import _pickle as cPickle
import os
import shutil
import math



def get_matrix():
    """
    获取特征向量和标签
    """
    
    feature_matrix=cPickle.load(open('data_0.pickle','rb'))
    origin_labels=list(feature_matrix[-1])
    feature_matrix=np.asarray(feature_matrix[:-1])
    feature_matrix=feature_matrix.reshape(500,2048)
    for i in range(1,96):
        temp_feature_matrix=cPickle.load(open('data_{}.pickle'.format(i),'rb'))
        origin_labels.extend(list(temp_feature_matrix[-1]))
        temp_feature_matrix=np.asarray(temp_feature_matrix[:-1])
        temp_feature_matrix=temp_feature_matrix.reshape(500,2048)
    
        feature_matrix=np.vstack((feature_matrix,temp_feature_matrix))
    return feature_matrix,origin_labels
    

def move(src,dst,i):
    """
    移动同一个聚类的表情包到同一个文件夹中
    """
    if not os.path.isdir(dst):
        os.makedirs(dst)
    s=src+'\\{}.jpg'.format(i)
    d=dst+'\\{}.jpg'.format(i)
    shutil.copy(s,d)



def cluster():
    """
    Kmeans做图像聚类
    """
    threshold=0.5
    num_clusters=20
    model=KMeans(n_clusters=num_clusters, max_iter=30000, tol=1e-15,n_init=400, \
                    init='k-means++',algorithm='full', n_jobs=-1)
    feature_matrix,origin_labels=get_matrix()
    model.fit(feature_matrix)
    result=model.predict(feature_matrix)
    for j in range(num_clusters):
    
        dis_total={}
        for i in range(len(origin_labels)):    
            if result[i]==j:
                dis=feature_matrix[i]-model.cluster_centers_[j]
                dis_2=math.sqrt(sum([k*k for k in dis]))
                dis_total[str(i)]=dis_2
        print("已经得到第{}个类别各个图片特征向量与聚类中心的欧式距离!".format(j))
        # 去除离群点，并将同一类图片移至同一个文件夹内
        for i in range(len(origin_labels)):
            if result[i]==j and dis_total[str(i)]<np.median(list(dis_total.values()))+threshold:
                move('..\Reptile\BQBCollection','{}'.format(j),origin_labels[i])

        print("已经成功将第{}个类别图片放到相应的文件夹内！".format(j))


if __name__=='__main__':
    cluster()
