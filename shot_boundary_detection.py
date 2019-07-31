import numpy as np
import cv2
import os
import errno
import skvideo.io
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from moviepy.editor import VideoFileClip
from utils import Shot

video_title = "ted_black_holes.mp4"
cap = cv2.VideoCapture(video_title)
fps = cap.get(cv2.CAP_PROP_FPS)
previous_frame = None
t_emd = 30
t_emd_diff = 20
bins = 64
shots = list()
counter = 0
frame_ctr = 0
first_frame = 1
MAX_SAMPLED = 10
sampled = 0


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8((bins-1) * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[int(img[i, j]/(256/bins))]
    return normalized.astype(int)


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * bins
    for i in range(h):
        for j in range(w):
            hist[int(img[i, j]/(256/bins))] += 1
    return np.array(hist) / (h * w)


def get_img(img):
    '''
    Prepare an image for image processing tasks
    '''
    # flatten returns a 2d grayscale array
    img = img.astype(int)
    img = normalize_exposure(img)
    return img


def earth_movers_distance(img1, img2):
    '''
    Measure the Earth Mover's distance between two images
    @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
    @returns:
    TODO
    '''
    img_a = get_img(img1)
    img_b = get_img(img2)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)


def hists_dist(img1, img2):
    img_a = get_img(img1)
    img_b = get_img(img2)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return euclidean(hist_a, hist_b)


def video_shot(frames, file_name):
    frames = np.expand_dims(frames, axis=-1)
    output_data = np.asarray(frames)
    output_data = output_data.astype(np.uint8)
    skvideo.io.vwrite(file_name, output_data)


stab = list()
time = 0

while True:
    # Capture frame-by-frame
    ret, orig_frame = cap.read()
    curr_shot = None
    if ret == False:
        break
    # Our operations on the frame come here
    b, g, r = cv2.split(orig_frame)
    # frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    processed_frame_b = cv2.resize(b, dsize=None, fx=0.4, fy=0.4)
    processed_frame_g = cv2.resize(g, dsize=None, fx=0.4, fy=0.4)
    processed_frame_r = cv2.resize(r, dsize=None, fx=0.4, fy=0.4)
    if first_frame == 1:
        # previous_frame_b = np.zeros(processed_frame_b.shape)
        # previous_frame_g = np.zeros(processed_frame_g.shape)
        # previous_frame_r = np.zeros(processed_frame_r.shape)
        previous_frame_b = processed_frame_b
        previous_frame_g = processed_frame_g
        previous_frame_r = processed_frame_r
        prev_emd_b = 0
        prev_emd_r = 0
        prev_emd_g = 0
        first_frame = 0
        curr_shot = Shot(0, ' ')
        shots.append(curr_shot)
    # Display the resulting frame
    # cv2.imshow('frame', orig_frame)
    # Applying SBD algorithm
    # EMD Distance from this paper - http://leibniz.cs.huji.ac.il/tr/1143.pdf
    if sampled == MAX_SAMPLED:
        # emd = earth_movers_distance(previous_frame, processed_frame)
        N = 8
        win_w = int(processed_frame_b.shape[0] / N)
        win_h = int(processed_frame_b.shape[1] / N)
        i = 0
        # Crop out the window and process
        emds_b = []
        emds_g = []
        emds_r = []
        for r in range(0, processed_frame_b.shape[0], win_w):
            for c in range(0, processed_frame_b.shape[1] - win_h, win_h):
                window1_b = previous_frame_b[r:r + win_w, c:c + win_h]
                window1_g = previous_frame_g[r:r + win_w, c:c + win_h]
                window1_r = previous_frame_r[r:r + win_w, c:c + win_h]
                window2_b = processed_frame_b[r:r + win_w, c:c + win_h]
                window2_g = processed_frame_g[r:r + win_w, c:c + win_h]
                window2_r = processed_frame_r[r:r + win_w, c:c + win_h]
                emd_b = hists_dist(window1_b, window2_b)
                emd_g = hists_dist(window1_g, window2_g)
                emd_r = hists_dist(window1_r, window2_r)
                emds_b.append(emd_b)
                emds_g.append(emd_g)
                emds_r.append(emd_r)
        sum_b = sum(emds_b)
        sum_g = sum(emds_g)
        sum_r = sum(emds_r)
        # print('B emds sum: ' + str(sum_b))
        # print('G emds sum: ' + str(sum_g))
        # print('R emds sum: ' + str(sum_r))
        # edge_diff = edge_based_difference(previous_frame, frame)
        time = float(frame_ctr / fps)
        sums_diff_b = np.linalg.norm(sum_b - prev_emd_b)
        sums_diff_r = np.linalg.norm(sum_r - prev_emd_r)
        sums_diff_g = np.linalg.norm(sum_g - prev_emd_g)
        # print('Sum difference of b: ' + str(sums_diff_b))
        # print('Sum difference of r: ' + str(sums_diff_r))
        # print('Sum difference of g: ' + str(sums_diff_g))
        if max([sum_b, sum_g, sum_r]) > t_emd and max([sums_diff_b, sums_diff_r, sums_diff_g]) > t_emd_diff:
        # if max([sums_diff_b, sums_diff_r, sums_diff_g]) > t_emd_diff:
            print("Transition has been detected at " + "t = " + str(time) + " seconds")
            if time - shots[-1].starting_time > 1:
                counter += 1
                curr_shot = Shot(time, ' ')
                shots[-1].ending_time = time
                print('starting time: ' + str(shots[-1].starting_time) +
                      ' ending time: ' + str(shots[-1].ending_time))
                shots.append(curr_shot)
        shots[-1].ending_time = time
        previous_frame_b = processed_frame_b
        previous_frame_g = processed_frame_g
        previous_frame_r = processed_frame_r
        prev_emd_b = sum_b
        prev_emd_r = sum_r
        prev_emd_g = sum_g
        sampled = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_ctr += 1
    sampled += 1
    shots[-1].add_frame(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY))

# For the last shot
shots[-1].ending_time = float(frame_ctr / fps)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(str(counter) + " shots has been detected")
print("number of shots is: " + str(len(shots)))

# path = os.getcwd() + '\shots'
path = os.getcwd() + '/shots'

try:
    os.makedirs(path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for i in range(len(shots)):
    curr_shot = shots[i]
    starting_time = curr_shot.starting_time
    ending_time = curr_shot.ending_time
    # ffmpeg_extract_subclip(video_title, starting_time, ending_time, targetname='shot ' + str(i) + '.mp4')
    print('starting time: ' + str(starting_time) + ' ending time: ' + str(ending_time))
    clip = VideoFileClip(video_title).subclip(starting_time, ending_time)
    clip.write_videofile('shots/shot_' + str(i) + '.mp4')
    clip.close()
    print('--------Shot ' + str(i) + ' has been saved------------')

print('--------Finish------------')
