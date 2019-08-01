import cv2
import numpy as np
import errno
from utils import Shot
from face_detection import detect_face
from ocr import TextualData
from ocr import collect_textual_data_for_frame
import os
import glob

POSITIVE = 1
NEGATIVE = -1
UNKNOWN = 0

theta_l = 2
theta_s = 10
theta_f = 0.3
theta_t = 4


def calc_shot_stab(shot):
    cap = cv2.VideoCapture(shot.title)
    while True:
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if ret == False:
            break
        # Our operations on the frame come here
        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        shot.add_frame(frame)
    shot_stability = shot.shot_stability_calc()
    return shot_stability


def calc_shot_textual_data(shot):
    i = 0
    cap = cv2.VideoCapture(shot.title)
    total = 0
    while True:
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if ret == False:
            break
        # Our operations on the frame come here
        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        td = collect_textual_data_for_frame(frame)
        words_num = td.words_num
        total += words_num
        i += 1
    if len == 0:
        return 0
    else:
        return td, total/i


def files_count():
    file_count = 0
    WorkingPath = os.path.dirname(os.path.abspath(__file__))
    for file in glob.glob(os.path.join(WorkingPath, '*.*')):
        file_count += 1
    return file_count


def classify_all_shots():
    os.chdir('shots')
    files_cnt = files_count()
    # path_POSITIVE = os.getcwd() + '\SOS'
    path_POSITIVE = os.getcwd() + '/SOS'
    try:
        os.makedirs(path_POSITIVE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path_NEGATIVE = os.getcwd() + '\SONS'
    path_NEGATIVE = os.getcwd() + '/SONS'
    try:
        os.makedirs(path_NEGATIVE)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path_UNKNOWN = os.getcwd() + '\UNKNOWN'
    path_UNKNOWN = os.getcwd() + '/UNKNOWN'
    try:
        os.makedirs(path_UNKNOWN)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for i in range(0, files_cnt):
        # video_title = "shots\shot_" + str(i) + ".mp4"
        video_title = "shot_" + str(i) + ".mp4"
        shot = Shot(0, video_title)
        print('Shot ' + str(i) + ' data:')
        shot_stab = calc_shot_stab(shot)
        print("shot stability: " + str(shot_stab))
        face_score = detect_face(shot)
        shot.face_score = face_score
        print("face score: " + str(face_score))
        _, f_t = calc_shot_textual_data(shot)
        shot.f_t = f_t
        print("Average number of words: " + str(f_t))
        shot.classify_shot(theta_l, theta_s, theta_f, theta_t)
        if shot.shot_type == POSITIVE:
            os.rename(video_title, path_POSITIVE + "/shot_" + str(i) + ".mp4")
        elif shot.shot_type == NEGATIVE:
            os.rename(video_title, path_NEGATIVE + "/shot_" + str(i) + ".mp4")
        else:
            os.rename(video_title, path_UNKNOWN + "/shot_" + str(i) + ".mp4")


classify_all_shots()





