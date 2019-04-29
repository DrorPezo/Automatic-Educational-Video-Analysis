import numpy as np
import cv2
import os
from skimage.io import imsave
import skvideo.io
from scipy.stats import wasserstein_distance

cap = cv2.VideoCapture("ted_black_holes.mp4")
previous_frame = None
t_emd = 0.0008
bins = 64
shots = list()
counter = 0
frame_ctr = 0
sampled = 0
MAX_SAMPLED = 100
first_frame = 1
fps = 30


class Shot:
    def __init__(self, t):
        self.starting_time = t
        self.frames_arr = list()

    def add_frame(self, img):
        self.frames_arr.append(img)


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


def edge_based_difference(img1, img2):
    tau = 0.1
    N = 8
    win_w = int(img1.shape[0]/N)
    win_h = int(img1.shape[1]/N)
    i = 0
    # Crop out the window and process
    for r in range(0, img1.shape[0], win_w):
        for c in range(0, img1.shape[1] - win_h, win_h):
            window1 = img1[r:r + win_w, c:c + win_h]
            edges1 = cv2.Canny(window1, 100, 200)
            window2 = img2[r:r + win_w, c:c + win_h]
            edges2 = cv2.Canny(window2, 100, 200)
            d = np.linalg.norm(edges2 - edges1)
            if d > tau:
                i += 1
    return i


def video_shot(frames, file_name):
    frames = np.expand_dims(frames, axis=-1)
    print(frames.shape)
    output_data = np.asarray(frames)
    output_data = output_data.astype(np.uint8)
    skvideo.io.vwrite(file_name, output_data)


while cap.isOpened():
    # Capture frame-by-frame
    ret, orig_frame = cap.read()
    if ret == False:
        break
    # Our operations on the frame come here
    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.resize(frame, dsize=None, fx=0.1, fy=0.1)
    if first_frame == 1:
        previous_frame = frame
        # imsave('shot_number_0.jpg', frame)
        first_frame = 0
        shots.append(Shot(0))

    # Display the resulting frame
    cv2.imshow('frame', processed_frame)
    # Applying SBD algorithm
    # EMD Distance from this paper - http://leibniz.cs.huji.ac.il/tr/1143.pdf
    if sampled == MAX_SAMPLED:
        emd = earth_movers_distance(previous_frame, processed_frame)
        # print(emd)
        # edge_diff = edge_based_difference(previous_frame, frame)
        if emd > t_emd:
            time = frame_ctr/fps
            print("Boundary shot has been detected " + "t = " + str(time) + " seconds")
            counter += 1
            shots.append(Shot(time))
            # imsave('shot_number_' + str(counter) + '.jpg', frame)
        previous_frame = processed_frame
        sampled = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    sampled += 1
    frame_ctr += 1
    shots[-1].add_frame(processed_frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(str(counter) + " shots has been detected")
# print("number of shots is: " + str(len(shots)))

# for i in range(len(shots)):
#     video_shot(shots[i].frames_arr, 'shot ' + str(i) + '.mp4')
#     print('--------Shot ' + str(i) + ' has been saved------------')

print('--------Finish------------')
