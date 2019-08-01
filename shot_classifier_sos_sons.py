import numpy as np
import cv2


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


shot_descs = shot_desc_arr('shot 89.mp4')
print(shot_descs.shape)


