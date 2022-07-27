#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import numpy as np
#读取和写入大量的图像数据
import imageio


#计算梯度能量
def energy(img):
    #原图转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #利用Sobel算子进行图像梯度计算
    #分别计算x方向和y方向的一阶导数
    x = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
    y = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
    #根据公式可知，能量值为x方向和y方向的一阶梯度的绝对值之和
    return cv2.add(np.absolute(x), np.absolute(y))


#从垂直方向上计算每点的能量
def Calculate_Horizontal_Energy(energy,img_gt):
    #图像高度和宽度
    height, width = energy.shape[:2]
    #用于存储每点能量
    energy_map = np.zeros((height, width))

    #若当前坐标在前景范围内则设置其能量为1e6
    for i in range(height):
        for j in range(width):
            if img_gt[i,j,0] > 200:
                energy_map[i,j] = 1e6
    
    #根据Forward Seam Removing机制计算能量
    for i in range(1, height):
        for j in range(width):
            up, down = (i-1) % height, (i+1) % height
            left, right = (j-1) % width, (j+1) % width
            if img_gt[up,j,0] == 0:
                cost_middle = np.abs(energy[i,right] - energy[i,left])
            else:
                cost_middle = 1e6
            if img_gt[up,left,0] == 0:
                cost_left = np.abs(energy[up,j] - energy[i,left]) + cost_middle
            else:
                cost_left = 1e6
            if img_gt[up,right,0] == 0:
                cost_right = np.abs(energy[up,j] - energy[i,right]) + cost_middle
            else:
                cost_right = 1e6
            #根据公式找出能量最小的组合，并更新能量图
            cost = np.array([cost_middle, cost_left, cost_right])
            M = cost + np.array([energy_map[up][j], energy_map[up][left], energy_map[up][right]])
            min_index = np.argmin(M)
            energy_map[i,j] = M[min_index]
            
    return energy_map


#连接垂直方向最小能量线
def Find_Vertical_EnergyLine(img,energy_map):
    #图像高度和宽度
    height, width = energy_map.shape[0], energy_map.shape[1]
     #当前行的最小能量值所在位置(height)
    cur = 0 
    SeamLine = []
    #从最后一行开始往前回溯
    for i in range(height - 1, -1, -1):
        #current_row存储当前所在行的所有元素(每个元素存储该点的能量值)
        current_row = energy_map[i, :]
        #如果处于最后一行，则直接找出最低能量值所在位置即可
        #argmin:找出最小值的下标
        if i == height - 1:
            cur = np.argmin(current_row)
        #若不在最后一行，则需要从上一行最小能量值的八连通区域中找出能量最小值
        #需要筛选的八连通区域为上一行最小能量值的上边一行(即当前行)的左、中、右三个位置
        else:
            #更新左中右三个位置的能量值
            if cur - 1 >= 0:
                left = current_row[cur - 1]
            else:
                left = 1e6
                
            middle = current_row[cur]
            
            if cur + 1 <= width - 1:
                right = current_row[cur + 1]
            else:
                right = 1e6
                
             #比较三者大小，根据最小值来对当前列的最小能量值的位置进行更新
            if left == min(left, middle, right):
                if cur == 0:
                    cur = 0
                else:
                    cur = cur - 1
            if middle == min(left, middle, right):
                cur = cur
            if right == min(left, middle, right):
                if cur == width - 1:
                    cur = cur
                else:
                    cur = cur + 1
        SeamLine.append([cur, i])
    return SeamLine


#移除垂直方向最小seam线
def Remove_Vertical_EnergyLine(img, img_gt,seam):
    #移除原图的Seam线
    height, width, depth = img.shape
    removed_img = np.zeros((height, width - 1, depth), np.uint8)
    for (y, x) in seam:
        removed_img[x, 0:y] = img[x, 0:y]
        removed_img[x, y:width - 1] = img[x, y + 1:width]
    #在移除原图的Seam线时，对前景图img_gt也要进行同样的操作
    height, width, depth = img_gt.shape
    removed_gt = np.zeros((height, width - 1, depth), np.uint8)
    for (y, x) in seam:
        removed_gt[x, 0:y] = img_gt[x, 0:y]
        removed_gt[x, y:width - 1] = img_gt[x, y + 1:width]
    return removed_img,removed_gt


#实时显示过程并将这一帧存入list方便之后生成GIF(针对垂直方向画线)
def Plot_Vertical(img, seam,image_list):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    image_list.append(img)
    cv2.imshow('Seam Carving', img)
    cv2.waitKey(1)
    
    
#实时显示过程并将这一帧存入list方便之后生成GIF(针对水平方向画线)   
def Plot_Horizontal(img, seam,image_list):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    image_list.append(img.transpose((1, 0, 2)))
    cv2.imshow('Seam Carving', img.transpose((1, 0, 2)))
    cv2.waitKey(1)


#主函数
if __name__ == "__main__":
    #图片编号，可修改
    img_number="15"
    
    #读取原图和前景图
    img = cv2.imread("C:/Users/93508/Desktop/Final/data/imgs/" + img_number + ".png")
    img_gt = cv2.imread("C:/Users/93508/Desktop/Final/data/gt/" + img_number + ".png")
    #存储原图和前景图的尺寸
    img_height, img_width = img.shape[0], img.shape[1]
    img_gt_height, img_gt_width = img_gt.shape[0], img_gt.shape[1]
    
    #压缩比例，可修改
    ratio = 0.8

    #压缩后的原图尺寸
    width = int(ratio * img_width)
    height = int(img_height * ratio)
    
    # 在目标目录保存原图和前景图用于对比
    cv2.namedWindow('Seam Carving', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Seam Carving', 500, 500)
    cv2.imwrite("C:/Users/93508/Desktop/Final/data/result/SeamCarving/" + img_number + "_origin.png", img)
    cv2.imwrite("C:/Users/93508/Desktop/Final/data/result/SeamCarving/" + img_number + "_gt.png", img_gt)
    
    # col_cnt：需要删除的列数
    # row_cnt：需要删除的行数    
    col_cnt = img_width - width
    row_cnt = img_height - height
    
    #用于存放生成GIF的各帧图片
    image_list=[]
    
    #先在垂直方向进行Seam Carving
    for i in range(col_cnt):
        energy_map = Calculate_Horizontal_Energy(energy(img),img_gt)
        SeamLine = Find_Vertical_EnergyLine(img,energy_map)
        Plot_Vertical(img, SeamLine,image_list)
        img,img_gt = Remove_Vertical_EnergyLine(img, img_gt,SeamLine)
        
    #对原图和前景图进行转置，即可继续调用垂直方向上的相关函数
    #水平方向进行Seam Carving
    img = img.transpose((1, 0, 2))
    img_gt = img_gt.transpose((1, 0, 2))
    for i in range(row_cnt):
        energy_map = Calculate_Horizontal_Energy(energy(img),img_gt)
        SeamLine = Find_Vertical_EnergyLine(img,energy_map)
        Plot_Horizontal(img, SeamLine,image_list)
        img,img_gt = Remove_Vertical_EnergyLine(img, img_gt,SeamLine)
    #最终保存的图片需要再转置回来
    img = img.transpose((1, 0, 2))
    
    #保存结果
    cv2.imwrite("C:/Users/93508/Desktop/Final/data/result/SeamCarving/" + img_number + '_result.png', img)
    cv2.imshow('Seam Carving', img)
    #保存GIF
    imageio.mimsave("C:/Users/93508/Desktop/Final/data/result/SeamCarving/" + img_number + '.gif', image_list, 'GIF', duration=0.1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

