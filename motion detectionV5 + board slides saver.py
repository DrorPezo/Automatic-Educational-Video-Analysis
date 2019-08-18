# -*- coding: utf-8 -*-

import cv2
import time
import datetime
import imutils
import numpy as np
import queue

colorCompArr=np.zeros([8,3])
edgesCompArr=np.zeros([8,3])
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


def EdgesCompMean(edgesCompCurr,win):
    global colorCompPos
    global edgesCompArr
    
    edgesCompArr[win,colorCompPos]=edgesCompCurr

    return np.mean(edgesCompArr[win,:])


def PreProcess(Window):
    hsv = cv2.cvtColor(Window, cv2.COLOR_BGR2HSV)
    blur_win = cv2.blur(hsv, (5,5))
    win_quantized=np.round(blur_win[:,:,2]/32)*32
    win_quantized=np.asarray(win_quantized, dtype="uint8" )
    return win_quantized


def motion_detection():
    global colorCompPos
    video_capture = cv2.VideoCapture('censor.mp4') 
    time.sleep(2)
    video_capture.set(1, 3000-1)
#    res, frame = cap.read()
    first_edges = None # initinate the first fame
    while True:
        frame = video_capture.read()[1] # gives 2 outputs retval,frame - [1] selects frame
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
        edges = cv2.Canny(frame, 100, 200)
        if first_edges is None:
            first_edges = edges 
            m, n=np.size(first_edges,0),np.size(first_edges,1)
            N=8
            win_w = int(m / N)
            win_h = int(n / N)
        else:
            pass
        edgesCompVals=np.zeros(N)
        colorCompPos=colorCompPos%3
        for c in range(0, N):
            left=np.multiply(c,win_h)
            right=np.multiply((c+1),win_h)
            if right == n:
                right=n-1
            window1 = first_edges[:, left:right]
            window2 = edges[:, left:right]
#            quantized_win1=PreProcess(window1)
#            quantized_win2=PreProcess(window2)
            grads1=np.sum(window1)
            grads2=np.sum(window2)
            if grads1/grads2>1:
                edgesRatio=grads1/grads2
            else:
                edgesRatio=grads2/grads1
            edgesRatioComp=EdgesCompMean(edgesRatio,c)
#            delta = colorsComp(quantized_win1, quantized_win2)
#            colorsCompCurr=colorCompMean(delta,c)
            edgesCompVals=np.append(edgesRatioComp,edgesCompVals)

        colorCompPos=colorCompPos+1   
        maxEMD = np.max(edgesCompVals)
#        frame_delta = colorsComp(first_edges, frame_quantized) 
#        colorComp=colorCompMean(frame_delta)

        if maxEMD>1.005:
            text = 'Moving'            
        else:
            pass

    
        cv2.putText(frame, 'Status: %s' % (text), 
            (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 255, 0), 4)
#        cv2.putText(frame, 'ColorComp:%s' % (colorComp), 
#            (10,60), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0), 4)
        frame_num=video_capture.get(1)
        cv2.putText(frame, 'Frame Num:%s' % (frame_num), 
            (10,100), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 0), 4)

        cv2.imshow('Security Feed', frame)
        cv2.imshow('equalization', edges)
        first_edges=edges
        key = cv2.waitKey(1) & 0xFF # (1) = time delay in seconds before execution, and 0xFF takes the last 8 bit to check value or sumin
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__=='__main__': 
    
    motion_detection()
