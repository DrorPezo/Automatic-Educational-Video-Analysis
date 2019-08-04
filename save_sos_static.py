import numpy as np
import cv2
import os
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
    slides = []
    prev_slide = None
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            sampled = 0
            slides.append(frame)

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
    return slide


os.chdir('shots')
os.chdir('SOS')
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


