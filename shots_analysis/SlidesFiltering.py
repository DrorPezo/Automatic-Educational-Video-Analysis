# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import os
from shots_analysis.motion_detectionV6 import motion_detection


def varCalc(img, channel):
    hist3 = cv2.calcHist([img], [channel], None, [256], [0, 256])
    value, counts = np.unique(img[:, :, 2], return_counts=True)
    maxCounts = counts.argsort()[-5:]
    hist3 = hist3[maxCounts]
    var3 = np.var(hist3)
    return var3/(np.size(img[:, :, 0])**2)


def slide_filtering(video_path):
    motion_detection(video_path)
    i = 0
    for filename in glob.glob('./slides board detected/Slide_*.png'): #assuming png

        img = cv2.imread(filename)

    #    cv2.imread( "./slides/croppedSlide_" + str(i) + ".png", board_img )
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if i % 2 == 0:
            oddHist = varCalc(hsv, 2)
    #        print(i,':',oddHist)
        else:
            pareHist = varCalc(hsv, 2)
    #        print(i,':',pareHist)
        if i > 0:
    #        print(i,' factor:',max(oddHist,pareHist)/min(oddHist,pareHist))
            if max(oddHist, pareHist)/min(oddHist, pareHist) < 4:
                os.remove(filename)

        i = i + 1
