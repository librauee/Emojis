# Emojis
Simple cluster of emojis, using machine learning,deep learning.

* 机器学习课程作业（自命题）

* 利用深度学习神经网络模型提取图片特征，以及机器学习聚类算法做特征的聚类

## Overview

整个工程可以分为三个模块:

* 第一个模块是网络爬虫部分，在知乎的表情包相关问答中爬取了 50655 张表情包，这里我采用了一个简单的分布式爬虫，爬取
表情包的链接与通过链接下载图片这两个过程同步进行，下载这一部分利用了多线程进行下载的加速，另外还包含一些简单的数据预处理的过程;
* 第二个模块是特征提取部分，利用 Inception v3 已经训练好的网络结构，对表情包做特征提取;
* 第三个模块是表情包聚类部分，利用 K-means 算法做表情包的聚类。

![流程图](https://github.com/librauee/Emojis/blob/master/flow.jpg)

## Result

![result1](https://github.com/librauee/Emojis/blob/master/1.jpg)
![result2](https://github.com/librauee/Emojis/blob/master/2.jpg)


[More](https://mp.weixin.qq.com/s/cS3BAt6Ej7qp22OttUxF4w)
