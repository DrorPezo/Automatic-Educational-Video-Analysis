# import required packages
import cv2
import dlib

cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
MAX_SAMPLED = 10


def detect_face(shot):
    cap = cv2.VideoCapture(shot.title)
    scores = []
    sampled = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = cnn_face_detector(gray, 1)
            for i, d in enumerate(dets):
                # print("Confidence: {}".format(d.confidence))
                scores.append(d.confidence)
            sampled = 0
        sampled += 1
    if scores:
        return max(scores)
    else:
        return 0

