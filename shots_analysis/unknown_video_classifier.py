import numpy as np
import cv2
import errno
from shots_analysis.sons_classifier import MIN_THRESH
import os
import re
from moviepy.editor import ImageSequenceClip


MIN_CHANGES = 20
MAX_DELTA = 120
MAX_SAMPLED = 5
ZERO_LEN_RATIO_MAX = 0.12
STATIC = 0
ANIMATION = 1
UNKNOWN = -1
MAX_MATCHES_RATIO = 9.5
path_ANIMATION = 'ANIMATION'


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
    fps = cap.get(cv2.CAP_PROP_FPS)
    threshes = []
    slides = []
    prev_slide = None
    frames_num = 0
    x0 = 0
    x1 = 0
    y0 = 0
    y1 = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
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
                    if prev_slide is None:
                        prev_slide = slide
                    black_img = np.zeros((400, 700, 3))
                    dist_from_black_img = np.linalg.norm(slide - black_img)
                    if dist_from_black_img > 5000:
                        # slides.append(slide)
                        # cv2.imshow('slide', slide)
                        # prev_slide = slide
                        break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cap = cv2.VideoCapture(video_title)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        slide = frame[y0:y1, x0:x1]
        # cv2.imshow('slide', slide)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        slides.append(slide)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # cv2.destroyAllWindows()
    return fps, slides[10:len(slides) - 10]


def classify_unknown(fps, slides):
    prev_slide = slides[0]
    i = 0
    matches_arr = []
    processed_slides = []
    for slide in slides:
        processed_slides.append(slide)
        matches = calculate_matches(slide, prev_slide)
        prev_slide = slide
        i += 1
        matches_arr.append(matches)
    buffer = []
    mtchs = []
    idx = []
    prev_match = 0
    ratio = 0
    i = 0
    lengths = []

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
    mtchs.append(buffer.copy())

    prev = None
    for i in range(0, len(mtchs)):
        l = mtchs[i]
        if len(l) == 1:
            if prev and i < len(mtchs) - 1:
                curr_val = l[0]
                prev_val = prev[-1]
                next_val = mtchs[i+1][0]
                dist_left = np.linalg.norm(curr_val - prev_val)
                dist_right = np.linalg.norm(curr_val - next_val)
                if dist_left < dist_right:
                    prev.append(curr_val)
                else:
                    mtchs[i+1].insert(0, curr_val)
        prev = l

    while True:
        flag = 1
        for l in mtchs:
            if len(l) == 1:
                mtchs.remove(l)
                flag = 0
        if flag == 1:
            break
    # print(mtchs)

    i = 0
    for m in mtchs:
        l = len(m)
        lengths.append(l)
        interval = [i, i + l]
        idx.append(interval)
        i += l

    union_idx = []
    buffer = idx[0]
    # print(idx)
    for l in idx:
        start = l[0]
        end = l[1]
        interval = end - start
        if interval < 5 * fps - 10:
            buffer[1] = end
        else:
            union_idx.append(buffer.copy())
            buffer[0] = start
            buffer[1] = end
    union_idx.append(buffer.copy())
    print(union_idx)
    for i in range(0, len(union_idx)):
        sub_shot = union_idx[i]
        start = sub_shot[0]
        end = sub_shot[1]
        video_path = "shot_" + str(shot_num) + "sub_shot_" + str(i) + ".mp4"
        print(sub_shot)
        clip = ImageSequenceClip(slides[start:end - 5], fps=fps)
        clip.write_videofile(video_path)
        clip.close()



os.chdir('SLIDES')

arr = os.listdir()

for file in arr:
    temp = re.findall(r'\d+', file)
    res = list(map(int, temp))
    if res:
        shot_num = res[0]
        print(file)
        fps, slides = extract_slides(file)
        classify_unknown(fps, slides)
