# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2
import numpy as np
img = cv2.imread('slidePics/15.png')
#img = cv2.imread('slidePics/4.png')


def line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det((line1[0], line1[2]), (line1[1], line1[3])), det((line2[0], line2[2]), (line2[1], line2[3])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
#    x=1
#    y=1
    return x, y


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
def gradCalc(edges,line,orient,side):
    m,n=np.size(edges,0), np.size(edges,1)
    if orient=='ver':
        up=np.min([line[0,1],line[0,3]])
        down=np.max([line[0,1],line[0,3]])
        if side=='left':
            grads=edges[up:down,line[0,0]:line[0,0]+round(n/20)]
        if side=='right':
            grads=edges[up:down,line[0,0]-round(n/20):line[0,0]]
    if orient=='hor':
        left=np.min([line[0,0],line[0,2]])
        right=np.max([line[0,0],line[0,2]])
        if side=='up':
            grads=edges[line[0,1]:line[0,1]+round(m/20),left:right]
        if side=='down':
            grads=edges[line[0,1]-round(m/20):line[0,1],left:right]
    return np.sum(grads)

img_quantized=np.round(img/8)*8
img_quantized=np.asarray(img_quantized, dtype="uint8" )
#cv2.imshow("img_quantized", img_quantized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
edges = cv2.Canny(img_quantized, 100, 200)
left,right=crop_img(edges)
cropped_edges=edges[:,left:right]
cropped_img=img[:,left:right]
cropped_quantized_img=img_quantized[:,left:right]


#bb, bg, br = cv2.split(cropped_img)
#plt.hist(bb.ravel(), 256, [0, 256])
#plt.hist(bg.ravel(), 256, [0, 256])
#plt.hist(br.ravel(), 256, [0, 256])
#plt.show()

m, n= np.size(cropped_img,0), np.size(cropped_img,1)
#cv2.imshow("Edges", cropped_edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 150, minLineLength=n/3, maxLineGap = n/6)
min_dhor_grad=np.inf
min_thor_grad=np.inf
min_rver_grad=np.inf
min_lver_grad=np.inf
xy_dhor = np.zeros(4)
xy_thor = np.zeros(4)
xy_rver = np.zeros(4)
xy_lver = np.zeros(4)

for line in lines:
        x1, y1, x2, y2 = line[0]
#        cv2.line(cropped_img, (x1, y1), (x2, y2), (255, 60, 0), 3)
        currLen=np.linalg.norm([x1-x2, y1-y2])
        

        if abs((y2-y1)/(x2-x1+0.01))>m/n:# vertical line
            if(x2+x1)/2>(2*n/3) and gradCalc(cropped_edges,line,'ver','right')<min_rver_grad:
                # line in right third
                min_rver_grad = gradCalc(cropped_edges,line,'ver','right')
                xy_rver = line[0]
            if(x2+x1)/2<(n/3) and gradCalc(cropped_edges,line,'ver','left')<min_lver_grad:# line in left third
                min_lver_grad = gradCalc(cropped_edges,line,'ver','left')
                xy_lver = line[0]
        if abs((y2-y1)/(x2-x1+0.01))<m/n:# horizion line
            if(y2+y1)/2<(m/3) and gradCalc(cropped_edges,line,'hor','up')<min_thor_grad:# top third
                min_thor_grad = gradCalc(cropped_edges,line,'hor','up')
                xy_thor = line[0]
            if(y2+y1)/2>(2*m/3) and gradCalc(cropped_edges,line,'hor','down')<min_dhor_grad:# down third
                min_dhor_grad = gradCalc(cropped_edges,line,'hor','down')
                xy_dhor = line[0]


    
p1 = line_intersection(xy_lver, xy_thor)

p2 = line_intersection(xy_lver, xy_dhor)
p3 = line_intersection(xy_rver, xy_thor)
p4 = line_intersection(xy_rver, xy_dhor)
p1=np.asarray(p1, dtype="int" )

p4=np.asarray(p4, dtype="int" )
if np.sum(xy_lver+xy_rver+xy_rver+xy_dhor)==0:
   p1=np.array([0,0])
   p4=np.array([n-1,m-1])   
else: 
    if np.sum(p1)==-2:
        if np.sum(xy_lver+xy_thor)==0:
            if np.sum(xy_dhor)!=0:
                p1[0]=np.min([xy_dhor[0],xy_dhor[2]])
            else:
                p1[0]=0
            if np.sum(xy_rver)!=0:
                p1[1]=np.min([xy_rver[1],xy_rver[3]])
            else:
                p1[1]=0
        elif np.sum(xy_lver)==0 and np.sum(xy_thor)!=0:
            p1[0]=np.min([xy_thor[0], xy_thor[2]])
            if p1[0]>n/3:
                p1[0]=0
            p1[1]=np.min([xy_thor[1], xy_thor[3]])
        elif np.sum(xy_lver)!=0 and np.sum(xy_thor)==0:
            p1[0]=np.min([xy_lver[0], xy_lver[2]])
            p1[1]=np.min([xy_lver[1], xy_lver[3]])
            if p1[1]>m/3:
                p1[1]=0
    if np.sum(p4)==-2:
        if np.sum(xy_rver+xy_dhor)==0:
            if np.sum(xy_thor)!=0:
                p4[0]=np.max([xy_thor[0],xy_thor[2]])
            else:
                p4[0]=n-1
            if np.sum(xy_lver)!=0:
                p4[1]=np.max([xy_lver[1],xy_lver[3]])
            else:
                p4[1]=m-1
        elif np.sum(xy_rver)==0 and np.sum(xy_dhor)!=0:
            p4[0]=np.max([xy_dhor[0], xy_dhor[2]])
            if p4[0]<2*n/3:
                p4[0]=n-1
            p4[1]=np.max([xy_dhor[1], xy_dhor[3]])
        elif np.sum(xy_rver)!=0 and np.sum(xy_dhor)==0:
            p4[0]=np.max([xy_rver[0], xy_rver[2]])
            p4[1]=np.max([xy_rver[1], xy_rver[3]])
            if p4[1]<2*m/3:
                p4[0]=m-1

cv2.rectangle(cropped_img,(p1[0],p1[1]), (p4[0],p4[1]),(0, 0, 255),5)

cv2.imshow("board", cropped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(cropped_img)
plt.show()
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#cropped_gray=gray[:,left:right]
#corners = cv2.cornerHarris(cropped_gray,2,3,0.002)
#plt.subplot(2,1,1), plt.imshow(corners ,cmap = 'jet')
#plt.title('Harris Corner Detection'), plt.xticks([]),
#plt.yticks([])
#corners2 = cv2.dilate(corners, None, iterations=3)
#cropped_quantized_img[corners2>0.01*corners2.max()] = [255,0,0]
#plt.subplot(2,1,2),plt.imshow(cropped_quantized_img,cmap = 'gray')
#plt.title('Canny Edge Detection'), plt.xticks([]),
#plt.yticks([])
#plt.show()