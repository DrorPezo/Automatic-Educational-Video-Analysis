import numpy as np
import cv2
import os
from skimage.io import imsave
from scipy.stats import wasserstein_distance

cap = cv2.VideoCapture("ted_black_holes.mp4")
previous_frame = None
t_emd = 0.0008
shots = []


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
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

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # Applying SBD algorithm
    # EMD Distance from this paper - http://leibniz.cs.huji.ac.il/tr/1143.pdf
    if previous_frame is not None:
        emd = earth_movers_distance(previous_frame, frame)
        # edge_diff = edge_based_difference(previous_frame, frame)
        if emd > t_emd:
            print("Boundary shot has been detected")
            counter += 1
            shots.append(frame)
            imsave('shot_number_' + str(counter) + '.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    previous_frame = frame

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(str(counter) + " shots has been detected")
