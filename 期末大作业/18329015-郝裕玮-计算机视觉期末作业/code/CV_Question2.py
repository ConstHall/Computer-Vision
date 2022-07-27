#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import cv2
import numpy as np


#并查集
class DisjointSet:
    #初始化
    def __init__(self, num_vertices):
        '''
            num_sets:簇的数量
            num_vertices:像素点数量
            data:每个簇自身所带的特征信息
        '''
        self.num_sets = num_vertices
        self.num_vertices = num_vertices
        self.data = np.empty((num_vertices, 4), dtype=np.int_)

        #像素点特征信息初始化
        for i in range(num_vertices):
            '''
                rank:合并时的从属关系(谁合并谁)
                size:每个簇的大小
                parent:用于判断属于哪个簇
                gt_num:簇内前景点的数量
            '''
            self.data[i, 0] = 0 # rank
            self.data[i, 1] = 1 # size
            self.data[i, 2] = i # parent
            self.data[i, 3] = 0 # gt_num

    #判断当前像素点所属的簇
    #并查集中的find操作
    def find(self, id):
        #递归回溯寻找所属的簇
        parent = id
        while parent != self.data[parent, 2]:
            parent = self.data[parent, 2]
        self.data[id, 2] = parent
        return parent

    #并查集中的join操作
    def join(self, id1, id2):
        #若合并的从属关系为：左>右，则右边的簇合并到左边
        if self.data[id1, 0] > self.data[id2, 0]:
            #合并需将左边的簇的像素点数量更新为二者之和
            #同时将右边的簇所属的父节点id修改为左边的
            self.data[id1, 1] += self.data[id2, 1]
            self.data[id2, 2] = id1
        #同理，将合并的关系调换即可
        else:
            self.data[id2, 1] += self.data[id1, 1]
            self.data[id1, 2] = id2
        self.num_sets -= 1


#计算两个像素点之间的距离
def dist(p1, p2):
    diff = pow((p1-p2),2)
    return np.sqrt(np.sum(diff))


#建图
def build_graph(img):
    #存储图的尺寸
    height, width = img.shape[:2]

    #edges：存储边
    #edges_dis：存储边的长度
    edges = []
    edges_dis = []

    #开始遍历整张图片建图
    for i in range(height):
        for j in range(width):
            #对当前点向正右方的点连接一条边
            if j+1 <= width-1:
                edges.append(np.array([i*width+j, i*width+(j+1)]))
                edges_dis.append(dist(img[i][j], img[i][j+1]))
            #对当前点向正下方的点连接一条边
            if i+1 <= height-1:
                edges.append(np.array([i*width+j, (i+1)*width+j]))
                edges_dis.append(dist(img[i][j], img[i+1][j]))

    #将数组设置为垂直向下存储每条边
    edges = np.vstack(edges).astype(int)
    edges_dis = np.array(edges_dis).astype(float)
    #考虑到我们需要根据边的大小来对edges和edges_dis两个数组进行相同的排序
    #而edges本身只存储边，无法用大小来进行排序
    #所以我们使用argsort来提取edges_dis数组从大到小排序的下标数组，再应用到edges数组中返回
    id = np.argsort(edges_dis)

    return edges[id], edges_dis[id]


#对图像进行分割
def segmentation(img, img_gt, k, min_num):
    #存储图片尺寸
    height, width = img.shape[:2]
    #分割之前先对该图像建立无向图
    edges, edges_dis = build_graph(img)
    
    #生成并查集
    djs = DisjointSet(height * width)
    threshold = np.zeros(height * width, dtype=float)
    #根据参考文献设立阈值函数
    for i in range(height * width):
        threshold[i] = k

    #开始合并
    for i in range(len(edges)):
        v1_parent = djs.find(edges[i, 0])
        v2_parent = djs.find(edges[i, 1])
        #若某条边的两点不属于同一个簇
        if v1_parent != v2_parent:
            #且不满足参考文献的判定标准
            if (edges_dis[i] <= threshold[v1_parent]) and (edges_dis[i] <= threshold[v2_parent]):
                #则对这两个簇进行合并
                djs.join(v1_parent, v2_parent)
                v1_parent = djs.find(v1_parent)
                #更新阈值函数
                threshold[v1_parent] = edges_dis[i] + k / djs.data[v1_parent, 1]

    #对较小的簇也进行合并
    while True:
        flag = True
        for i in range(len(edges)):
            v1_parent = djs.find(edges[i, 0])
            v2_parent = djs.find(edges[i, 1])
            #若该边的两个点不属于同一个簇且其中一个簇的像素点数量小于阈值50
            #则对这两个簇进行合并
            if (v1_parent != v2_parent) and ((djs.data[v1_parent, 1] < min_num) or (djs.data[v2_parent, 1] < min_num)):
                flag = False
                djs.join(v1_parent, v2_parent)
        if flag:
            break
    
    #区域标记
    for i in range(height):
        for j in range(width):
            #若当前像素点位置为白色(前景区域)
            #则为该像素点所属的簇对其内部属性gt_num = gt_num + 1
            if img_gt[i][j] > 200:
                djs.data[djs.find(i*width+j), 3] += 1
    
    #根据之前的区域标记生成我们自己前景图
    res = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            parent = djs.find(i*width+j)
            #若当前像素点所属的簇内部gt_num超过一半，则设置当前点为白色(对于所有这个簇的像素点都会变为白色)
            #即 gt_num / size >= 0.5
            if djs.data[parent, 3] / djs.data[parent, 1] >= 0.5:
                res[i][j] = 255
    return res, djs


#主函数
if __name__ == "__main__":
    #需要遍历的文件名
    a = ["15","115","215","315","415","515","615","715","815","915"]

    cnt = 0
    cnt2 = 0
    correct = 0
    # k代表每个簇的阈值
    k = 130

    #遍历测试图片
    for img_number in a:
        #每个簇的像素点数量不得少于min_num
        min_num = 50
        img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png",cv2.IMREAD_COLOR)
        img_gt = cv2.imread("C:/Users/93508/Desktop/Final/data/gt/" + img_number + ".png",cv2.IMREAD_GRAYSCALE)
        
        res, djs = segmentation(img, img_gt, k, min_num)

        #将前景图写入文件夹用于对比
        cv2.imwrite("C:/Users/93508/Desktop/Final/data/result/Segmentation/" + img_number + "_gt.png", img_gt)
        #将生成的前景图保存至对应目录
        smt_output_path = "C:/Users/93508/Desktop/Final/data/result/Segmentation"
        filename = img_number + "_result.png"
        cv2.imwrite(os.path.join(smt_output_path, filename), res)

        #保存前景图尺寸
        height, width = img_gt.shape
        #用于计算IOU
        intersection = 0
        union = 0

        #遍历前景图每个像素点
        for i in range(height):
            for j in range(width):
                #若生成图和前景图的当前像素点都是白色则 intersection + 1
                if img_gt[i][j] > 200 and res[i][j] > 200:
                    intersection += 1
                #若生成图和前景图的当前像素点不全是白色则 union + 1
                if img_gt[i][j] > 200 or res[i][j] > 200:
                    union += 1
        #统计合并后簇的数量
        cnt += djs.num_sets
        if djs.num_sets >= 50 and djs.num_sets <= 70:
            cnt2 = cnt2 + 1
        print("%s   %d   %.2f%%" %(filename, djs.num_sets, intersection / union*100))
        #用于统计平均正确率
        correct = correct + intersection / union
    print("\nK = %d\nAverage num_sets = %.1f\nCorrect num_sets = %d\nAverage Correctness = %.2f%%\n" %(k, cnt/10, cnt2, correct/10*100))


# In[ ]:




