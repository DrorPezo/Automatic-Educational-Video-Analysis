import cv2
import numpy as np
import errno
from shot_utils import Shot
import re
import os
import glob
from shot_utils import is_video_file

POSITIVE = 1
NEGATIVE = -1
UNKNOWN = 0

theta_l = 2
theta_s = 11.5
theta_f = 0.3
theta_t = 3.5
theta_c = 2.5
theta_ct = 60
MAX_SAMPLED = 10


def calc_shot_stab(shot, rect_points):
    cap = cv2.VideoCapture(shot.title)
    print(shot.title)
    if rect_points:
        print(rect_points)
        up_left = rect_points[0].tolist()
        down_right = rect_points[1].tolist()
        w = np.linalg.norm(up_left[0] - down_right[0])
        h = np.linalg.norm(up_left[1] - down_right[1])
    while True:
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if ret == False:
            break
        if rect_points is not None:
            orig_frame = orig_frame[up_left[1]: int(up_left[1] + h), up_left[0]:int(up_left[0] + w)]
        # Our operations on the frame come here
        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        shot.add_frame(frame)
    shot_stability = shot.shot_stability_calc()
    return shot_stability


def files_count():
    file_count = 0
    WorkingPath = os.getcwd()
    for file in glob.glob(os.path.join(WorkingPath, '*.mp4*')):
        file_count += 1
    return file_count


def classify_all_shots(video_title, rect_points):
    shots = []
    vt = os.path.splitext(video_title)[0]
    path = 'shots'

    os.chdir(path)
    if rect_points is not None:
        os.chdir('SONS')

    path_POSITIVE = 'SOS'
    try:
        os.makedirs(path_POSITIVE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    path_NEGATIVE = 'SONS'
    try:
        os.makedirs(path_NEGATIVE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    path_UNKNOWN = 'UNKNOWN'
    try:
        os.makedirs(path_UNKNOWN)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    arr = os.listdir()

    for file in arr:
        if is_video_file(file):
            temp = re.findall(r'\d+', file)
            res = list(map(int, temp))
            if res:
                shot_num = res[0]
                video_title = "shot_" + str(shot_num) + ".mp4"
                shot = Shot(0, video_title)
                print('Shot ' + str(shot_num))
                shot_stab = calc_shot_stab(shot, rect_points)
            shot.classify_shot(theta_l, theta_s, theta_c, theta_ct)
            shots.append(shot)
            if shot.shot_type == POSITIVE:
                path = os.path.join(path_POSITIVE, "shot_" + str(shot_num) + ".mp4")
                os.rename(video_title, path)
            elif shot.shot_type == NEGATIVE:
                path = os.path.join(path_NEGATIVE, "shot_" + str(shot_num) + ".mp4")
                os.rename(video_title, path)
            else:
                path = os.path.join(path_UNKNOWN, "shot_" + str(shot_num) + ".mp4")
                os.rename(video_title, path)







