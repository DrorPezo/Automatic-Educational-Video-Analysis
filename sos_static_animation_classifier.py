import numpy as np
import cv2
import os
from utils import edge_based_difference
import re
import errno

MIN_CHANGES = 20
MAX_DELTA = 500
MAX_SAMPLED = 5
ZERO_LEN_RATIO_MAX = 0.12
STATIC = 0
ANIMATION = 1
UNKNOWN = -1
MAX_MATCHES_RATIO = 9.5
MIN_THRESH = 50


def calculate_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if des1 is None or des2 is None:
        return 0
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [0 for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = 1

    return sum(matchesMask)


def classify_static_animation(video_title):
    cap = cv2.VideoCapture(video_title)
    slides = []
    slide_type = -1
    sampled = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            slides.append(frame)
            sampled = 0
        sampled += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
            if delta > MAX_DELTA or ratio > 25:
                mtchs.append(buffer.copy())
                buffer.clear()
        buffer.append(match)
        prev_match = match
        i += 1
    mtchs.append(buffer.copy())
    lengths_ratios = []
    sz = 0
    for l in mtchs:
        sz += len(l)
    for l in mtchs:
        lsz = len(l)
        lengths_ratios.append(lsz/sz)
    lengths_ratios = [i * 100 for i in lengths_ratios]
    length_ratio_hist, bin_edges = np.histogram(lengths_ratios, bins=range(0, 101, 20))
    matches_hist, bin_edges = np.histogram(matches_arr, bins=5)
    if matches_hist[0] != 0:
        matches_ratio = matches_hist[-1]/matches_hist[0]
    else:
        matches_ratio = MAX_MATCHES_RATIO + 1
    if matches_ratio > MAX_MATCHES_RATIO:
        print(lengths_ratios)
        print(length_ratio_hist)
        if len(lengths_ratios) < 2 and length_ratio_hist[2] > 0:
            slide_type = STATIC
            print('static slide')
        else:
            # print(length_ratio_hist)
            if length_ratio_hist[2] > 0 or length_ratio_hist[3] > 0:
                # SBD to the shot according the matches array
                slide_type = UNKNOWN
                print('unknown')
            else:
                index = np.argmax(np.array(matches_arr))
                slide = processed_slides[index]
                slide_type = STATIC
                print('static slide')
    else:
        print(lengths_ratios)
        print(length_ratio_hist)
        if len(lengths_ratios) < 2:
            slide_type = ANIMATION
            print('animation')
        else:
            if length_ratio_hist[2] > 0 or length_ratio_hist[3] > 0 or length_ratio_hist[4] > 0:
                # need to perform another SBD to the shot according the matches array
                slide_type = UNKNOWN
                print('unknown')
            else:
                slide_type = ANIMATION
                print('animation')

    cap.release()
    cv2.destroyAllWindows()
    return slide_type


os.chdir('shots')
os.chdir('SOS')

path_ANIMATION = 'ANIMATION'
try:
    os.mkdir(path_ANIMATION)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

path_STATIC = 'STATIC'
try:
    os.mkdir(path_STATIC)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

path_UNKNOWN = 'UNKNOWN'
try:
    os.mkdir(path_UNKNOWN)
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
        slide_type = classify_static_animation(file)
        if slide_type == STATIC:
            os.path.join(path_STATIC, "shot_" + str(shot_num) + ".mp4")
            # print("shot_" + str(shot_num) + ".mp4")
            os.rename(file, path_STATIC + "\shot_" + str(shot_num) + ".mp4")
        elif slide_type == ANIMATION:
            os.path.join(path_ANIMATION, "shot_" + str(shot_num) + ".mp4")
            # print("shot_" + str(shot_num) + ".mp4")
            os.rename(file, path_ANIMATION + "\shot_" + str(shot_num) + ".mp4")
        elif slide_type == UNKNOWN:
            os.path.join(path_UNKNOWN, "shot_" + str(shot_num) + ".mp4")
            # print("shot_" + str(shot_num) + ".mp4")
            os.rename(file, path_UNKNOWN + "\shot_" + str(shot_num) + ".mp4")

