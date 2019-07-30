import cv2
import numpy as np
from utils import Shot
from face_detection import detect_face
from ocr import TextualData
from ocr import collect_textual_data_for_frame

video_title = "shot 77.mp4"
shot = Shot(0)
len = shot.calculate_len()

def calc_shot_stab(video_title):
    cap = cv2.VideoCapture(video_title)
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


def calc_shot_textual_data(video_title):
    cap = cv2.VideoCapture(video_title)
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
    return td, total/len


shot_stab = calc_shot_stab(video_title)
_, face_score = detect_face(video_title)
td, f_t = calc_shot_textual_data(video_title)





