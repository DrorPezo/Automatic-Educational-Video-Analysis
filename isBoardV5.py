# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:46:57 2019

@author: someone
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D

def entropy1(labels, base=2):  
  value,counts = np.unique(labels, return_counts=True)
  maxCounts=counts.argsort()[-10:-5]
  return entropy(counts[maxCounts], base=base)

def colorsComp(img,i):

#    hist = cv2.calcHist([img], [0, 1, 2], None, [1, 256, 3],[0, 256, 0, 256, 0, 256])    
    hist1 = cv2.calcHist([img], [0], None, [256],[0, 256])
    
#    value,counts = np.unique(img[:,:,0], return_counts=True)
#    maxCounts=counts.argsort()[-20:-3]
#    hist1=hist1[maxCounts]
#    hist1=np.asarray(hist1, dtype="uint8" )
    hist2 = cv2.calcHist([img], [1], None, [256],[0, 256])
    
#    value,counts = np.unique(img[:,:,1], return_counts=True)
#    maxCounts=counts.argsort()[-20:-3]
#    hist2=hist2[maxCounts]
#    hist2=np.asarray(hist2, dtype="uint8" )
    hist3 = cv2.calcHist([img], [2], None, [256],[0, 256])
    
#    value,counts = np.unique(img[:,:,2], return_counts=True)
#    maxCounts=counts.argsort()[-20:-3]
#    hist3=hist3[maxCounts]
#    hist3=np.asarray(hist3, dtype="uint8" )
#    c1=np.mean([np.correlate(hist1[:,0],hist2[:,0])])
#    c2=np.mean([np.correlate(hist1[:,0],hist3[:,0])])
#    c3=np.mean([np.correlate(hist2[:,0],hist2[:,0])])
    
#    return np.mean([c1,c2,c3])/(np.size(img[:,:,0])**2)
    return np.mean([cv2.compareHist(hist1,hist2,method=i),cv2.compareHist(hist1,hist3,method=i),cv2.compareHist(hist2,hist3,method=i)])/(np.size(img[:,:,0]))
def crop_img(edges):
    n=np.size(edges,1)
    for i in range(0, n):
        if sum(edges[:,i])>0:
            left=i
            break
    for j in range(0, n):        
        if sum(edges[:,n-1-j])>0:
            right=n-1-j
            break
    return left,right

def varCalc(img,channel):
    hist3 = cv2.calcHist([img], [channel], None, [256],[0, 256]) 
    value,counts = np.unique(img[:,:,2], return_counts=True)
    maxCounts=counts.argsort()[-5:]
    hist3=hist3[maxCounts]
    var3=np.var(hist3)
    return var3/(np.size(img[:,:,0])**2)

def winCheck(img):
    
    colorsCompVal=colorsComp(img,1)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(bgr_img)
    
    b_ent=entropy1(b)
    g_ent=entropy1(g)
    r_ent=entropy1(r)
    arr=np.array([b_ent,g_ent,r_ent])
    var3=varCalc(img,2)

    
#    ax.scatter( colorsCompVal,var3,np.mean(arr))

    if (var3>0.04) or np.mean(arr)<0.01 or colorsCompVal>1000:
        return -1
    else:
        return 1

    



def isBoard(img):
    m, n=np.size(img,0),np.size(img,1)
    N=8
    win_w = int(m / N)
    win_h = int(n / N)
    false=0

    for r in range(0, N):
        for c in range(0, N):
            left=np.multiply(c,win_h)
            right=np.multiply((c+1),win_h)
            top=np.multiply(r,win_w)
            down=np.multiply((r+1),win_w)
            if right == n:
                right=n-1
            window = img[top:down, left:right]
            if winCheck(window)==-1:
               false=1        
#    ax.set_xlabel("var value")
#    ax.set_ylabel("hist comparsion")
#    ax.set_zlabel("mean entropy")
    if false==1:
        return -1
    return 1

# print('slides')
# for i in range (1, 8):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     simg = cv2.imread("slidePics/" + str(i) + ".png")
#
#
#     edges=cv2.Canny(simg, 100, 200)
#     left,right=crop_img(edges)
#     cropped_edges=edges[:,left:right]
#     cropped_img=simg[:,left:right]
#     cropped_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
#
#     plt.title("Slide " + str(i))
#     plt.show()
#     if isBoard(cropped_hsv)==1:
#         print(i)
#
# print('boards')
# for i in range (2, 9):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     simg = cv2.imread("onlyBoardPics/" + str(i) + ".png")
#
#     edges=cv2.Canny(simg, 100, 200)
#     left,right=crop_img(edges)
#     cropped_edges=edges[:,left:right]
#     cropped_img=simg[:,left:right]
#     cropped_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
#
#     plt.title("Board " + str(i-1))
#     plt.show()
#     if isBoard(cropped_hsv)==1:
#         print(i)