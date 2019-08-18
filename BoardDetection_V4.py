# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2
import numpy as np
img = cv2.imread('slidePics/15.png')

def line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det((line1[0],line1[2]),(line1[1],line1[3])), det((line2[0],line2[2]),(line2[1],line2[3])))
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



img_quantized=np.round(img/8)*8
img_quantized=np.asarray(img_quantized, dtype="uint8" )
cv2.imshow("img_quantized", img_quantized)
cv2.waitKey(0)
cv2.destroyAllWindows()
edges = cv2.Canny(img_quantized, 100, 200)
left,right=crop_img(edges)
cropped_edges=edges[:,left:right]
cropped_img=img[:,left:right]
cropped_quantized_img=img_quantized[:,left:right]
#g_quantize=round(g/4)*4
#b_quantize=round(b/4)*4
#cv2.HoughLines()

bb, bg, br = cv2.split(cropped_img)
plt.hist(bb.ravel(), 256, [0, 256])
plt.hist(bg.ravel(), 256, [0, 256])
plt.hist(br.ravel(), 256, [0, 256])

plt.show()

m, n= np.size(cropped_img,0), np.size(cropped_img,1)
cv2.imshow("Edges", cropped_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 150, minLineLength=n/3, maxLineGap = n/6)
max_dhor_len=np.zeros(1)
max_thor_len=np.zeros(1)
max_rver_len=np.zeros(1)
max_lver_len=np.zeros(1)
xy_dhor = np.zeros(4)
xy_thor = np.zeros(4)
xy_rver = np.zeros(4)
xy_lver = np.zeros(4)

for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(cropped_img, (x1, y1), (x2, y2), (255, 60, 0), 3)
        currLen=np.linalg.norm([x1-x2, y1-y2])
#        lineLen = np.append(lineLen,currLen)
        
#        if abs((y2-y1)/(x2-x1+0.01))<m/n:
#            horLines=np.append(line[:],horLines,axis=0)
#            horLengths = np.append(currLen,horLengths,axis=0)
        if abs((y2-y1)/(x2-x1+0.01))>m/n:# anach
            if(x2+x1)/2>(2*n/3) and currLen > max_rver_len:# right third
                max_rver_len = currLen
                xy_rver = line[0]
            if(x2+x1)/2<(n/3) and currLen > max_lver_len:# left third
                max_lver_len = currLen
                xy_lver = line[0]
        if abs((y2-y1)/(x2-x1+0.01))<m/n:# ofki
            if(y2+y1)/2>(2*m/3) and currLen > max_thor_len:# top third
                max_thor_len = currLen
                xy_thor = line[0]
            if(y2+y1)/2<(m/3) and currLen > max_dhor_len:# down third
                max_dhor_len = currLen
                xy_dhor = line[0]
#maxHorlenghs=horLengths.argsort()[-3:]
#for i in horLines 

#if(sum(xy_tver)!=0&&sum(xy_rhor)!=0)
p1 = line_intersection(xy_lver, xy_thor)

p2 = line_intersection(xy_lver, xy_dhor)
p3 = line_intersection(xy_rver, xy_thor)
p4 = line_intersection(xy_rver, xy_dhor)
#p4=(p1[0]+max_thor_len,p1[1]+max_lver_len)
p1=np.asarray(p1, dtype="int" )
#p1=int(p1)
p4=np.asarray(p4, dtype="int" )
#cv2.line(cropped_img, (p1[0],p1[1]), (p4[0],p4[1]), (255, 60, 0), 3)
cv2.rectangle(cropped_img,(p1[0],p1[1]), (p4[0],p4[1]),(0, 0, 255),5)
#x=np.array([int(np.round(t1)),int(np.round(t2))])
#cv2.circle(cropped_img, (x[0],x[1]),5, (0, 0, 255), 3)
#cv2.line(cropped_img, (xy_hor[0], xy_hor[1]), (xy_hor[2], xy_hor[3]), (0, 255, 0), 3)
#cv2.line(cropped_img, (xy_ver[0], xy_ver[1]), (xy_ver[2], xy_ver[3]), (0, 255, 0), 3)
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