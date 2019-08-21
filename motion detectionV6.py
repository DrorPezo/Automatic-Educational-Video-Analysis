# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from scipy.stats import entropy
import os


path = "."
directory = "slides"
parent_dir = "./"
path = os.path.join(parent_dir, directory)
try:
    os.mkdir( path )
except:
    pass

startPose=100
colorCompArr=np.zeros([8,10])
colorCompPos=0
def colorsComp(img1,img2):
    hist1 = cv2.calcHist([img1], [0], None, [256],[0, 256])
    value,counts = np.unique(img1[:], return_counts=True)
    maxCounts=counts.argsort()[-4:]
    hist1=hist1[maxCounts]
    
    hist2 = cv2.calcHist([img2], [0], None, [256],[0, 256])
    value,counts = np.unique(img2[:], return_counts=True)
    maxCounts=counts.argsort()[-4:]
    hist2=hist2[maxCounts]

    return np.correlate(hist1[:,0],hist2[:,0])/(np.size(img1)**2)
def colorCompMean(colorCompCurr,win):
    global colorCompPos
    global colorCompArr    
    colorCompArr[win,colorCompPos]=colorCompCurr
    return np.mean(colorCompArr[win,:])

def PreProcess(Window):
    hsv = cv2.cvtColor(Window, cv2.COLOR_BGR2HSV)
    m,n=np.size(hsv,0),np.size(hsv,1)
    blur_win = cv2.blur(hsv, (int(m/72),int(n/16)))
    win_quantized=np.round(blur_win[:,:,2]/32)*32
    win_quantized=np.asarray(win_quantized, dtype="uint8" )
    return win_quantized

def entropy1(labels, base=2):  
  value,counts = np.unique(labels, return_counts=True)
  maxCounts=counts.argsort()[-20:-2]
  return entropy(counts[maxCounts], base=base)

def motion_detection():
    global colorCompPos
    video_capture = cv2.VideoCapture('censor.mp4') 
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('output2.avi',fourcc, 30.0, (480,360))
#    out=cv2.VideoWriter('video1.avi',cv2.CV_FOURCC('M','J','P','G'),32,(360,480),1)

    time.sleep(2)
    video_capture.set(1, startPose-1)
    first_frame = None 
    slideNum=0
    flag=-1
    framesNum=video_capture.get(7)
    while True:
        frame = video_capture.read()[1] # gives 2 outputs retval,frame - [1] selects frame
        frame_num=video_capture.get(1)
        text = 'Not Moving'
        black_frame=np.zeros([np.size(frame,0),np.size(frame,1)])
        if first_frame is None:
            first_frame = frame 
            m, n=np.size(first_frame,0),np.size(first_frame,1)
            N=8
            win_w = int(m / N)
            win_h = int(n / N)
        else:
            pass
        colorsCompVals=np.zeros(N)
        colorCompPos=colorCompPos%10
        for c in range(0, N):
            left=np.multiply(c,win_h)
            right=np.multiply((c+1),win_h)
            if right == n:
                right=n-1
            window1 = first_frame[:, left:right]
            window2 = frame[:, left:right]
            quantized_win1=PreProcess(window1)
            quantized_win2=PreProcess(window2)
            delta = colorsComp(quantized_win1, quantized_win2)
            colorsCompCurr=colorCompMean(delta,c)
            colorsCompVals=np.append(colorsCompCurr,colorsCompVals)

        colorCompPos=colorCompPos+1   
        maxEMD = np.max(colorsCompVals)

        if maxEMD>5e-06:
            text = 'Moving'
            flag=1
        else:
            pass
        
        if text == 'Not Moving' and flag==1 and frame_num>300:#first time not moving for current section
            flag=-1
            video_capture.set(1, frame_num-30)
            _, lastBoard = video_capture.read()
            if framesNum>frame_num+30:
                video_capture.set(1, frame_num+30)
                _, nextBoard = video_capture.read()
            video_capture.set(1, frame_num)
            last_edge=cv2.Canny(lastBoard, 100, 200)
            curr_edge=cv2.Canny(nextBoard, 100, 200)
            #second condition when video is going to be finished
            if np.sum(curr_edge)*1.2<np.sum(last_edge):
                cv2.imwrite( "./slides/slide_" + str(slideNum) + ".png", lastBoard )
                slideNum=slideNum+1
        
        if framesNum==frame_num+90:
            cv2.imwrite( "./slides/slide_" + str(slideNum) + ".png", frame )

        cv2.putText(frame, 'Status: %s' % (text), 
            (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 255, 0), 4)

        
        cv2.putText(frame, 'Frame Num:%s' % (frame_num), 
            (10,100), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 0), 4)
        
        cv2.putText(frame, 'Slide Num:%s' % (slideNum), 
            (10,140), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 255), 4)
        
        cv2.imshow('Feed', frame)
#        out.write(frame)
        
        first_frame=frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or framesNum==frame_num+90:
            video_capture.release()
#            out.release()
            cv2.destroyAllWindows()
            break
                


if __name__=='__main__': 
    
    motion_detection()
