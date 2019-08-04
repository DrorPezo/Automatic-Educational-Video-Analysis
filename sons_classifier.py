import numpy as np
import cv2
import errno
import os
import re

MAX_SAMPLED = 15
MAX_FAILS = 3
MIN_THRESH = 50


def calc_feat_desc(img):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def shot_desc_arr(video_title):
    cap = cv2.VideoCapture(video_title)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = calc_feat_desc(gray)
    des = np.asarray(des)
    return des


def classify_SONS(video_title):
    cap = cv2.VideoCapture(video_title)
    sampled = 0
    successes = 0
    threshes = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            sampled = 0
            frame_area = frame.shape[0] * frame.shape[1]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist[:, 0]
            hist = hist / sum(hist)
            thresh_val = np.average(range(0, 32),  weights=hist) * 8
            threshes.append(thresh_val)
            # check if image is dark
            if thresh_val < MIN_THRESH:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv += int(thresh_val)
                gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
            # cv2.imshow('slide', thresh)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, 3, (0, 255, 0), 3)
            max_contour = None
            max_contour_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_contour_area:
                    max_contour_area = area
                    max_contour = contour
            if max_contour is not None:
                if len(max_contour) > 4:
                    rect = cv2.minAreaRect(max_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    p1, p2, p3, p4 = (box[0], box[1], box[2], box[3])
                    x0 = min([box[0][0], box[2][0]])
                    x1 = max([box[0][0], box[2][0]])
                    y0 = min([box[0][1], box[1][1]])
                    y1 = max([box[0][1], box[1][1]])
                    slide = frame[y0:y1, x0:x1]
                    rect_area = slide.shape[0] * slide.shape[1]
                    if max_contour_area > 0.2 * frame_area and rect_area > 0.15 * frame_area:
                        if rect_area < 1.5 * max_contour_area:
                            # img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                            h_ind = (slide.shape[0] > 0.3 * frame.shape[0]) and (slide.shape[0] < 0.6 * frame.shape[0])
                            w_ind = (slide.shape[1] > 0.3 * frame.shape[1]) and (slide.shape[1] < 0.6 * frame.shape[1])
                            # cv2.imshow(video_title, img)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            # cv2.imwrite('slide of ' + video_title + '_' + str(i/16) + '.jpg', slide)
                            # print(h_ind)
                            # print(w_ind)
                            if h_ind and w_ind:
                                successes += 1
        sampled += 1
    return successes > MAX_FAILS

os.chdir('shots')
os.chdir('SONS')

path_SLIDE = os.getcwd() + '\SLIDES'
# path_SLIDE = os.getcwd() + '/SLIDES'
try:
    os.makedirs(path_SLIDE)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

path_NON_SLIDE = os.getcwd() + '\ANYTHING_ELSE'
# path_NON_SLIDE = os.getcwd() + '/NON SLIDES'
try:
    os.makedirs(path_NON_SLIDE)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

arr = os.listdir()
for file in arr:
    temp = re.findall(r'\d+', file)
    res = list(map(int, temp))
    if res:
        shot_num = res[0]

        if classify_SONS(file):
            print(path_SLIDE + "\shot_" + str(shot_num) + ".mp4")
            os.rename(file, path_SLIDE + "\shot_" + str(shot_num) + ".mp4")
        else:
            print(path_NON_SLIDE + "\shot_" + str(shot_num) + ".mp4")
            os.rename(file, path_NON_SLIDE + "\shot_" + str(shot_num) + ".mp4")









