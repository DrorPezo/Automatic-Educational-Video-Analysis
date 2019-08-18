# -*- coding: utf-8 -*-

import cv2
import time
import datetime
import imutils
import numpy as np
import queue
from scipy.stats import entropy

colorCompArr=np.zeros([8,10])
colorCompPos=0
def colorsComp(img1,img2):
#    hist = cv2.calcHist([img], [0, 1, 2], None, [1, 256, 3],[0, 256, 0, 256, 0, 256])    
    hist1 = cv2.calcHist([img1], [0], None, [256],[0, 256])
    value,counts = np.unique(img1[:], return_counts=True)
    maxCounts=counts.argsort()[-4:]
    hist1=hist1[maxCounts]
    
    hist2 = cv2.calcHist([img2], [0], None, [256],[0, 256])
    value,counts = np.unique(img2[:], return_counts=True)
    maxCounts=counts.argsort()[-4:]
    hist2=hist2[maxCounts]

    return np.correlate(hist1[:,0],hist2[:,0])/(np.size(img1)**2)
#    return np.correlate(hist1[:,0],hist2[:,0])/(np.size(img1)**2)
def colorCompMean(colorCompCurr,win):
    global colorCompPos
    global colorCompArr    
    colorCompArr[win,colorCompPos]=colorCompCurr
    return np.mean(colorCompArr[win,:])

def PreProcess(Window):
    hsv = cv2.cvtColor(Window, cv2.COLOR_BGR2HSV)
    blur_win = cv2.blur(hsv, (5,5))
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
    time.sleep(2)
    video_capture.set(1, 1-1)
    first_frame = None 
    slideNum=0
    flag=-1
    framesNum=video_capture.get(7)
    while True:
        frame = video_capture.read()[1] # gives 2 outputs retval,frame - [1] selects frame
        frame_num=video_capture.get(1)
        text = 'Not Moving'
        
#        partsComp=frameParts(frame)
#        relevant[:,:,2]=frame[:,:,2]
#        relevant[:,:,1]=frame[:,:,1]
#        relevant[:,:,0]=frame[:,:,0]
#        relevant=np.asarray(relevant, dtype="uint8" )
#        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#        blur_frame = cv2.blur(hsv, (5,5))
#        frame_quantized=np.round(blur_frame[:,:,2]/32)*32
#        frame_quantized=np.asarray(frame_quantized, dtype="uint8" )
#        gaussian_frame = cv2.GaussianBlur(frame_quantized, (21,21),0)
#        edges = cv2.Canny(frame, 100, 200)
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
#        frame_delta = colorsComp(first_frame, frame_quantized) 
#        colorComp=colorCompMean(frame_delta)

        if maxEMD>5e-06:
            text = 'Moving'
            flag=1
        else:
            pass
        
        if text == 'Not Moving' and flag==1 and frame_num>300:#first time not moving for current section
            flag=-1
            video_capture.set(1, frame_num-20)
            _, lastBoard = video_capture.read()
            if framesNum>frame_num+20:
                video_capture.set(1, frame_num+20)
                _, nextBoard = video_capture.read()
            video_capture.set(1, frame_num)
            last_edge=cv2.Canny(lastBoard, 100, 200)
            curr_edge=cv2.Canny(nextBoard, 100, 200)
            if np.sum(curr_edge)*1.2<np.sum(last_edge) or framesNum<frame_num+20:
                cv2.imwrite( "Image_" + str(slideNum) + ".png", lastBoard )
                slideNum=slideNum+1
                
        cv2.putText(frame, 'Status: %s' % (text), 
            (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 255, 0), 4)
#        cv2.putText(frame, 'ColorComp:%s' % (colorComp), 
#            (10,60), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0), 4)
        
        cv2.putText(frame, 'Frame Num:%s' % (frame_num), 
            (10,100), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 0), 4)
        
        cv2.putText(frame, 'Slide Num:%s' % (slideNum), 
            (10,140), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 255), 4)

        cv2.imshow('Security Feed', frame)
#        cv2.imshow('equalization', edges)
        
        first_frame=frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
                


if __name__=='__main__': 
    
    motion_detection()
