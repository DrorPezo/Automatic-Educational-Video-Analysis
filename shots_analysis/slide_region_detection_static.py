import numpy as np
import cv2
import os
from shots_analysis.sons_classifier import MIN_THRESH
from utils import edge_based_difference
import re
import errno

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
    changes = []
    dist_from_bi = []
    prev_slide = None
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
                        slides.append(slide)
                        if prev_slide is None:
                            prev_slide = slide
                        black_img = np.zeros((400, 700, 3))
                        dist_from_black_img = np.linalg.norm(slide - black_img)
                        if dist_from_black_img > 5000:
                            _, changes_cnt = edge_based_difference(slide, prev_slide)
                            changes.append(changes_cnt)
                            # cv2.imshow('slide', slide)
                            prev_slide = slide

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sampled += 1

    cap.release()
    prev_slide = slides[0]
    i = 0
    matches_arr = []
    processed_slides = []
    for slide in slides:
        if i % 3 == 0:
            i += 1
            continue
        processed_slides.append(slide)
        matches = calculate_matches(slide, prev_slide)
        prev_slide = slide
        i += 1
        matches_arr.append(matches)
    buffer = []
    mtchs = []
    prev_match = 0
    ratio = 0
    i = 0
    for match in matches_arr:
        delta = np.linalg.norm(match - prev_match)
        if prev_match != 0:
            ratio = match/prev_match
        if i != 0:
            if delta > MAX_DELTA or ratio > 30:
                mtchs.append(buffer.copy())
                buffer.clear()
        buffer.append(match)
        prev_match = match
        i += 1

    index = np.argmax(np.array(matches_arr))
    slide = processed_slides[index]
    slide = np.asarray(slide)
    h, w, c = slide.shape
    slide = slide[10:(h-10), 10:(w-10), :]
    return slide


def save_slide():
    os.chdir('SLIDES')
    os.chdir('STATIC')

    path_IMAGES = 'IMAGES'
    try:
        os.mkdir(path_IMAGES)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    arr = os.listdir()

    for file in arr:
        temp = re.findall(r'\d+', file)
        res = list(map(int, temp))
        if res:
            shot_num = res[0]
            print(file)
            slide = extract_slides(file)
            img_path = os.path.join(path_IMAGES, "slide_" + str(shot_num) + ".jpg")
            cv2.imwrite(img_path, slide)


