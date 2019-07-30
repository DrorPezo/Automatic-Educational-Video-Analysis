# import required packages
import cv2
import dlib

cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def detect_face(video_title):
    cap = cv2.VideoCapture(video_title)
    scores = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = cnn_face_detector(gray, 1)
        for i,d in enumerate(dets):
            print("Confidence: {}".format(d.confidence))
        scores.append(d)
        return scores, max(scores)

