import numpy as np
import cv2
import errno
import os
import re
from train_sbd import Rectangle

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


def tag_SONS(video_title):
    cap = cv2.VideoCapture(video_title)
    threshes = []
    rectangles = []
    sample = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_area = frame.shape[0] * frame.shape[1]
        if sample == MAX_SAMPLED:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist[:, 0]
            hist = hist / sum(hist)
            thresh_val = np.average(range(0, 32), weights=hist) * 8
            threshes.append(thresh_val)
            # check if image is dark

            ret, thresh = cv2.threshold(gray, thresh_val / 2, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, thresh_val, 255)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = None
            max_contour_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_contour_area:
                    max_contour_area = area
                    max_contour = contour
            if max_contour is not None:
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                p1, p2, p3, p4 = (box[0], box[1], box[2], box[3])
                rect = Rectangle(p2, p4)
                w = rect.w
                h = rect.h
                rect_area = rect.area
                w_ind = (w > 0.2 * frame.shape[1]) and (w < 0.9 * frame.shape[1])
                h_ind = (h > 0.2 * frame.shape[0]) and (h < 0.9 * frame.shape[0])
                area_ind = rect_area > 0.1 * frame_area
                if area_ind and w_ind and h_ind:
                    return True
            sample = 0
        sample += 1
    return False


def classify_sons(video_title):
    vt = os.path.splitext(video_title)[0]
    path = 'shots_' + vt
    os.chdir(path)
    os.chdir('SONS')

    path_SLIDE = 'SLIDES'
    try:
        os.makedirs(path_SLIDE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    path_NON_SLIDE = 'ANYTHING_ELSE'
    try:
        os.makedirs(path_NON_SLIDE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    arr = os.listdir()
    for file in arr:
        print(file)
        temp = re.findall(r'\d+', file)
        res = list(map(int, temp))
        if res:
            shot_num = res[0]
            if tag_SONS(file):
                path = os.path.join(path_SLIDE, "shot_" + str(shot_num) + ".mp4")
                print(path)
                os.rename(file, path)
                pass
            else:
                path = os.path.join(path_NON_SLIDE, "shot_" + str(shot_num) + ".mp4")
                print(path)
                os.rename(file, path)
                pass












