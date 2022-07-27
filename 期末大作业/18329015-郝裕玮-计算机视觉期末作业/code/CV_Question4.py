#!/usr/bin/env python
# coding: utf-8

# 问题1——手工实现PCA，特征向量降为16D

# In[31]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

#手工实现PCA
# X代表训练集矩阵
# k代表需要保留的特征数量（即降维后的特征向量数量）
def pca(X,k):
    temp = k
    #矩阵行数代表样本数量n_samples，矩阵列数代表每个样本的特征数n_features(即特征向量)
    n_samples, n_features = X.shape
    #对矩阵进行零均值化，即减去这一行的均值
    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X = X - mean

    #散度矩阵scatter_matrix
    #散度矩阵就是协方差矩阵*(总数据量-1),因此他们的特征根和特征向量是一样的
    scatter_matrix = np.dot(np.transpose(norm_X),norm_X)

    #计算特征向量和特征值
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]

    #根据特征值大小，从大到小对(特征值,特征向量)的多对pair进行排序
    eig_pairs.sort(reverse=True)
    #选出前k个特征向量
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    
    #输出降维后的特征向量
    print("降维后的特征向量为：\n")
    for k in range(16):
        for i in range(12):
            for j in range(12):
                print("%.3f  "%(feature[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")

    #将特征向量rescale到0——255之间
    array = feature
    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(16):
        for j in range(144):
            array[i][j] = int(round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin))
    print("rescale到0—255范围内的特征向量为：\n")
    for k in range(16):
        for i in range(12):
            for j in range(12):
                print("%d  "%(array[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")
    feature = np.array([ele[1] for ele in eig_pairs[:temp]])
    
    #Y = PX即为降维到k维后的数据
    data = np.dot(norm_X,np.transpose(feature))
    return data


#存储裁剪后图片的每个子图(12*12)
X = np.zeros((12,12),dtype = int)
#存储所有子图，组成训练集
X_new = np.zeros((144,144),dtype = int)

#读取代表图片
#img_num可修改为自己需要读取的图片编号
img_number = "915"
img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
#以灰度图的方式读取，去除图片的三通道RGB信息
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#将图片尺寸裁剪为12的倍数
img = cv2.resize(img,(144,144))

#将图片拆分12*12的子图并全部保存至X_new
#每个12*12的子图被视为144维的向量，将其一维扁平化存入X_new的每一行中
#由于图片为144*144，子图为12*12，所以X_new为144*144
#即144条数据，每条数据有144维的特征
for i in range(0,144):
    X = img[int(i/12)*12:int(i/12)*12+12,i%12*12:i%12*12+12]
    X_new[i,:] = X.reshape(1,-1)

#调用手工实现的PCA
result = pca(X_new,16)
print("压缩结果为：\n")
print(result)


# 问题1——调用Sklearn库的PCA函数，特征向量降为16D

# In[28]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


#存储裁剪后图片的每个子图(12*12)
X = np.zeros((12,12),dtype = int)
#存储所有子图，组成训练集
X_new = np.zeros((144,144),dtype = int)

#读取代表图片
#img_num可修改为自己需要读取的图片编号
img_number = "915"
img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
#以灰度图的方式读取，去除图片的三通道RGB信息
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#将图片尺寸裁剪为12的倍数
img = cv2.resize(img,(144,144))

#将图片拆分12*12的子图并全部保存至X_new
#每个12*12的子图被视为144维的向量，将其一维扁平化存入X_new的每一行中
#由于图片为144*144，子图为12*12，所以X_new为144*144
#即144条数据，每条数据有144维的特征
for i in range(0,144):
    X = img[int(i/12)*12:int(i/12)*12+12,i%12*12:i%12*12+12]
    X_new[i,:] = X.reshape(1,-1)


# 可修改n的数量来选择需要保留的特征数量（即降维后的结果）
n = 16
#调用sklearn库中的PCA
#n_components: 需要保留的特征数量（即降维后的结果）
pca = PCA(n_components = n) #实例化
newX = pca.fit_transform(X_new) #用已有数据训练PCA模型，并返回降维后的数据

#PCA中的属性components_为降维后的特征向量
print("降维后的特征向量为：\n")
for k in range(16):
    for i in range(12):
        for j in range(12):
            print("%.3f  "%(pca.components_[k][i*12+j]),end="")
        print("\n")
    print("\n")
print("-----------------------------------------------------------------------------")

#将特征向量rescale到0——255之间
array = pca.components_
ymax = 255
ymin = 0
xmax = max(map(max,array))
xmin = min(map(min,array))
for i in range(16):
    for j in range(144):
        array[i][j] = int(round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin))
print("rescale到0—255范围内的特征向量为：\n")
for k in range(16):
    for i in range(12):
        for j in range(12):
            print("%d  "%(array[k][i*12+j]),end="")
        print("\n")
    print("\n")
print("-----------------------------------------------------------------------------")

print("压缩结果为：\n")
print(newX)


# 问题2——144D

# In[26]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

#手工实现PCA
# X代表训练集矩阵
# k代表需要保留的特征数量（即降维后的特征向量数量）
def pca(X,k):
    temp = k
    #矩阵行数代表样本数量n_samples，矩阵列数代表每个样本的特征数n_features(即特征向量)
    n_samples, n_features = X.shape
    #对矩阵进行零均值化，即减去这一行的均值
    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X = X - mean

    #散度矩阵scatter_matrix
    #散度矩阵就是协方差矩阵*(总数据量-1),因此他们的特征根和特征向量是一样的
    scatter_matrix = np.dot(np.transpose(norm_X),norm_X)

    #计算特征向量和特征值
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]

    #根据特征值大小，从大到小对(特征值,特征向量)的多对pair进行排序
    eig_pairs.sort(reverse=True)
    #选出前k个特征向量
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    
    #输出降维后的特征向量
    print("降维后的特征向量为：\n")
    for k in range(144):
        for i in range(12):
            for j in range(12):
                print("%.3f  "%(feature[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")

    #将特征向量rescale到0——255之间
    array = feature
    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(144):
        for j in range(144):
            array[i][j] = int(round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin))
    print("rescale到0—255范围内的特征向量为：\n")
    for k in range(144):
        for i in range(12):
            for j in range(12):
                print("%d  "%(array[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")
    feature = np.array([ele[1] for ele in eig_pairs[:temp]])
    
    #Y = PX即为降维到k维后的数据
    data = np.dot(norm_X,np.transpose(feature))
    return data


#存储裁剪后图片的每个子图(12*12)
X = np.zeros((12,12),dtype = int)
#存储所有子图，组成训练集
X_new = np.zeros((144,144),dtype = int)

#读取代表图片
#img_num可修改为自己需要读取的图片编号
img_number = "915"
img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
#以灰度图的方式读取，去除图片的三通道RGB信息
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#将图片尺寸裁剪为12的倍数
img = cv2.resize(img,(144,144))

#将图片拆分12*12的子图并全部保存至X_new
#每个12*12的子图被视为144维的向量，将其一维扁平化存入X_new的每一行中
#由于图片为144*144，子图为12*12，所以X_new为144*144
#即144条数据，每条数据有144维的特征
for i in range(0,144):
    X = img[int(i/12)*12:int(i/12)*12+12,i%12*12:i%12*12+12]
    X_new[i,:] = X.reshape(1,-1)

#调用手工实现的PCA
result = pca(X_new,144)
print("压缩结果为：\n")
print(result)


# 问题2——60D

# In[24]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

#手工实现PCA
# X代表训练集矩阵
# k代表需要保留的特征数量（即降维后的特征向量数量）
def pca(X,k):
    temp = k
    #矩阵行数代表样本数量n_samples，矩阵列数代表每个样本的特征数n_features(即特征向量)
    n_samples, n_features = X.shape
    #对矩阵进行零均值化，即减去这一行的均值
    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X = X - mean

    #散度矩阵scatter_matrix
    #散度矩阵就是协方差矩阵*(总数据量-1),因此他们的特征根和特征向量是一样的
    scatter_matrix = np.dot(np.transpose(norm_X),norm_X)

    #计算特征向量和特征值
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]

    #根据特征值大小，从大到小对(特征值,特征向量)的多对pair进行排序
    eig_pairs.sort(reverse=True)
    #选出前k个特征向量
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    
    #输出降维后的特征向量
    print("降维后的特征向量为：\n")
    for k in range(60):
        for i in range(12):
            for j in range(12):
                print("%.3f  "%(feature[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")

    #将特征向量rescale到0——255之间
    array = feature
    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(60):
        for j in range(144):
            array[i][j] = int(round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin))
    print("rescale到0—255范围内的特征向量为：\n")
    for k in range(60):
        for i in range(12):
            for j in range(12):
                print("%d  "%(array[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")
    feature = np.array([ele[1] for ele in eig_pairs[:temp]])
    
    #Y = PX即为降维到k维后的数据
    data = np.dot(norm_X,np.transpose(feature))
    return data


#存储裁剪后图片的每个子图(12*12)
X = np.zeros((12,12),dtype = int)
#存储所有子图，组成训练集
X_new = np.zeros((144,144),dtype = int)

#读取代表图片
#img_num可修改为自己需要读取的图片编号
img_number = "915"
img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
#以灰度图的方式读取，去除图片的三通道RGB信息
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#将图片尺寸裁剪为12的倍数
img = cv2.resize(img,(144,144))

#将图片拆分12*12的子图并全部保存至X_new
#每个12*12的子图被视为144维的向量，将其一维扁平化存入X_new的每一行中
#由于图片为144*144，子图为12*12，所以X_new为144*144
#即144条数据，每条数据有144维的特征
for i in range(0,144):
    X = img[int(i/12)*12:int(i/12)*12+12,i%12*12:i%12*12+12]
    X_new[i,:] = X.reshape(1,-1)

#调用手工实现的PCA
result = pca(X_new,60)
print("压缩结果为：\n")
print(result)


# 问题2——6D

# In[22]:


import numpy as np
import os
import cv2
import shutil
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

#手工实现PCA
# X代表训练集矩阵
# k代表需要保留的特征数量（即降维后的特征向量数量）
def pca(X,k):
    temp = k
    #矩阵行数代表样本数量n_samples，矩阵列数代表每个样本的特征数n_features(即特征向量)
    n_samples, n_features = X.shape
    #对矩阵进行零均值化，即减去这一行的均值
    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X = X - mean

    #散度矩阵scatter_matrix
    #散度矩阵就是协方差矩阵*(总数据量-1),因此他们的特征根和特征向量是一样的
    scatter_matrix = np.dot(np.transpose(norm_X),norm_X)

    #计算特征向量和特征值
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]

    #根据特征值大小，从大到小对(特征值,特征向量)的多对pair进行排序
    eig_pairs.sort(reverse=True)
    #选出前k个特征向量
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    
    #输出降维后的特征向量
    print("降维后的特征向量为：\n")
    for k in range(6):
        for i in range(12):
            for j in range(12):
                print("%.3f  "%(feature[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")

    #将特征向量rescale到0——255之间
    array = feature
    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(6):
        for j in range(144):
            array[i][j] = int(round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin))
    print("rescale到0—255范围内的特征向量为：\n")
    for k in range(6):
        for i in range(12):
            for j in range(12):
                print("%d  "%(array[k][i*12+j]),end="")
            print("\n")
        print("\n")
    print("-----------------------------------------------------------------------------")
    
    feature = np.array([ele[1] for ele in eig_pairs[:temp]])
    
    #Y = PX即为降维到k维后的数据
    data = np.dot(norm_X,np.transpose(feature))
    return data


#存储裁剪后图片的每个子图(12*12)
X = np.zeros((12,12),dtype = int)
#存储所有子图，组成训练集
X_new = np.zeros((144,144),dtype = int)

#读取代表图片
#img_num可修改为自己需要读取的图片编号
img_number = "915"
img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
#以灰度图的方式读取，去除图片的三通道RGB信息
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#将图片尺寸裁剪为12的倍数
img = cv2.resize(img,(144,144))

#将图片拆分12*12的子图并全部保存至X_new
#每个12*12的子图被视为144维的向量，将其一维扁平化存入X_new的每一行中
#由于图片为144*144，子图为12*12，所以X_new为144*144
#即144条数据，每条数据有144维的特征
for i in range(0,144):
    X = img[int(i/12)*12:int(i/12)*12+12,i%12*12:i%12*12+12]
    X_new[i,:] = X.reshape(1,-1)

#调用手工实现的PCA
result = pca(X_new,6)
print("压缩结果为：\n")
print(result)


# In[ ]:




