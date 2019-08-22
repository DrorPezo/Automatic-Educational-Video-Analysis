import numpy as np
import cv2
import os
import re
import errno
from moviepy.editor import ImageSequenceClip


MIN_CHANGES = 20
MAX_DELTA = 500
MAX_SAMPLED = 5
ZERO_LEN_RATIO_MAX = 0.12
STATIC = 0
ANIMATION = 1
UNKNOWN = -1
MAX_MATCHES_RATIO = 9.5


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


def find_slide(video_title, rect_points):
    cap = cv2.VideoCapture(video_title)
    up_left = rect_points[0].tolist()
    down_right = rect_points[1].tolist()
    w = np.linalg.norm(up_left[0] - down_right[0])
    h = np.linalg.norm(up_left[1] - down_right[1])
    slides = []
    while True:
        ret, orig_frame = cap.read()
        if ret == False:
            break
        orig_frame = orig_frame[up_left[1]: int(up_left[1] + h), up_left[0]:int(up_left[0] + w)]
        slides.append(orig_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    return slides


def extract_slides(video_title, rect_points):
    vt = os.path.splitext(video_title)[0]
    path = 'shots_' + vt
    os.chdir(path)
    os.chdir('SONS')
    os.chdir('SLIDES')

    path_GIFS = 'EXTRACTED_SLIDES'
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
            up_left = rect_points[0].tolist()
            down_right = rect_points[1].tolist()
            w = np.linalg.norm(up_left[0] - down_right[0])
            h = np.linalg.norm(up_left[1] - down_right[1])
            stream = ffmpeg.input(file)
            ffmpeg.crop(stream, up_left[0], up_left[1], w, h)
            slides = find_slide(file, rect_points)
            video_path = os.path.join(path_GIFS, "slide_" + str(shot_num) + ".mp4")
            # cv2.imwrite(img_path, slide)
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            clip = ImageSequenceClip(slides, fps=fps)
            clip.write_videofile(video_path)
            clip.close()
            slides.clear()


