import numpy as np
import cv2
import os
from shots_analysis.sons_classifier import MIN_THRESH
import re
import errno
from moviepy.editor import ImageSequenceClip

MIN_CHANGES = 20
MAX_DELTA = 500
MAX_SAMPLED = 5


def calculate_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [0 for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = 1

    return sum(matchesMask)


def extract_slides(video_title):
    cap = cv2.VideoCapture(video_title)
    sampled = 0
    threshes = []
    slides = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            sampled = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist[:, 0]
            hist = hist / sum(hist)
            thresh_val = np.average(range(0, 32), weights=hist) * 8
            threshes.append(thresh_val)
            # check if image is dark
            if thresh_val < MIN_THRESH:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv += int(thresh_val)
                gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
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
                    x0 = min([box[0][0], box[2][0]])
                    x1 = max([box[0][0], box[2][0]])
                    y0 = min([box[0][1], box[1][1]])
                    y1 = max([box[0][1], box[1][1]])
                    slide = frame[y0:y1, x0:x1]
                    if slide.shape[0] > 0.1 * frame.shape[0] and slide.shape[1] > 0.1 * slide.shape[1]:
                        slide = cv2.resize(slide, (700, 400), interpolation=cv2.INTER_AREA)
                        black_img = np.zeros((400, 700, 3))
                        dist_from_black_img = np.linalg.norm(slide - black_img)
                        if dist_from_black_img > 5000:
                            slide = frame[y0:y1, x0:x1]
                            # cv2.imshow('slide', slide)
                            # print(str([y0, y1, x0, x1]))
                            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sampled += 1
    cap.release()
    cap = cv2.VideoCapture(video_title)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        slide = frame[y0:y1, x0:x1]
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        slides.append(slide)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return slides[10:len(slides) - 10]


def save_animation():
    os.chdir('SLIDES')
    os.chdir('ANIMATION')

    path_GIFS = 'GIFS'
    try:
        os.mkdir(path_GIFS)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    arr = os.listdir()

    for file in arr:
        temp = re.findall(r'\d+', file)
        res = list(map(int, temp))
        if res:
            shot_num = res[0]
            slides = extract_slides(file)
            video_path = os.path.join(path_GIFS, "slide_" + str(shot_num) + ".mp4")
            # cv2.imwrite(img_path, slide)
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            clip = ImageSequenceClip(slides, fps=fps)
            clip.write_videofile(video_path)
            clip.close()


