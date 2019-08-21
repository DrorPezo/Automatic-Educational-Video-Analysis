import numpy as np
import cv2
from scipy.spatial.distance import euclidean


MAX_SAMPLED = 10
MAX_FAILS = 3


class Rectangle:
    def __init__(self, up_left, down_right):
        self.up_left = up_left
        self.down_right = down_right
        self.w = np.linalg.norm(self.up_left[0] - self.down_right[0])
        self.h = np.linalg.norm(self.up_left[1] - self.down_right[1])
        self.area = self.w * self.h
        self.count = 0


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    bins = 32
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
    bins = 32
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


def hists_dist(img1, img2):
    img_a = get_img(img1)
    img_b = get_img(img2)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return euclidean(hist_a, hist_b)


def search_close_rectangle(rectangles, rect):
    for rectangle in rectangles:
        left_delta = np.linalg.norm(np.array(rect.up_left) - np.array(rectangle.up_left))
        right_delta = np.linalg.norm(np.array(rect.down_right) - np.array(rectangle.down_right))
        if left_delta < 50 and right_delta < 50:
            rectangle.count += 1
            return True
    return False


def train_sbd(video_title):
    cap = cv2.VideoCapture(video_title)
    threshes = []
    rectangles = []
    sample = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_area = frame.shape[0] * frame.shape[1]
        if sample == MAX_SAMPLED:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist[:, 0]
            hist = hist / sum(hist)
            thresh_val = np.average(range(0, 32), weights=hist) * 8
            threshes.append(thresh_val)
            # check if image is dark
            ret, thresh = cv2.threshold(gray, thresh_val/2, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, thresh_val, 255)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = None
            max_contour_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_contour_area:
                    max_contour_area = area
                    max_contour = contour
            if max_contour is not None:
                # approx = cv2.approxPolyDP(max_contour, 0.01 * cv2.arcLength(max_contour, True), True)
                # cv2.drawContours(frame, [approx], 0, (0,0,255), 5)
                # print(approx.ravel())
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                p1, p2, p3, p4 = (box[0], box[1], box[2], box[3])
                rect = Rectangle(p2, p4)
                w = rect.w
                h = rect.h
                rect_area = rect.area
                w_ind = (w > 0.2 * frame.shape[1]) and (w < 0.9 * frame.shape[1])
                h_ind = (h > 0.2 * frame.shape[0]) and (h < 0.9 * frame.shape[0])
                area_ind = rect_area > 0.1 * frame_area

                if area_ind and w_ind and h_ind:
                    # cv2.rectangle(frame, tuple(p2), tuple(p4), (0, 255, 0), 3)
                    if not search_close_rectangle(rectangles, rect):
                        rectangles.append(rect)
            sample = 0
        sample += 1
    #     cv2.imshow(video_title, frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    rectangles.sort(key=lambda rectangle: rectangle.count, reverse=True)
    return [rectangles[0].up_left, rectangles[0].down_right]


def train_thresh(video_title, rect_points):
    # these values should be selected by outliers detections by histogram of values
    threshes =[]
    cap = cv2.VideoCapture(video_title)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_frame = 1
    # 10 for short video with many animations. For longer video 150 is good
    MAX_SAMPLED = 10
    # MAX_SAMPLED = 150
    sampled = 0

    frame_ctr = 0
    up_left = rect_points[0].tolist()
    down_right = rect_points[1].tolist()
    w = np.linalg.norm(up_left[0] - down_right[0])
    h = np.linalg.norm(up_left[1] - down_right[1])
    while True:
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if ret == False:
            break
        orig_frame = orig_frame[up_left[1]: int(up_left[1] + h), up_left[0]:int(up_left[0] + w)]
        # Our operations on the frame come here
        b, g, r = cv2.split(orig_frame)
        processed_frame_b = cv2.resize(b, dsize=None, fx=0.4, fy=0.4)
        processed_frame_g = cv2.resize(g, dsize=None, fx=0.4, fy=0.4)
        processed_frame_r = cv2.resize(r, dsize=None, fx=0.4, fy=0.4)
        if first_frame == 1:
            previous_frame_b = processed_frame_b
            previous_frame_g = processed_frame_g
            previous_frame_r = processed_frame_r
            prev_emd_b = 0
            prev_emd_r = 0
            prev_emd_g = 0
            first_frame = 0

        # Applying SBD algorithm
        if sampled == MAX_SAMPLED:
            N = 8
            win_w = int(processed_frame_b.shape[0] / N)
            win_h = int(processed_frame_b.shape[1] / N)
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
            time = float(frame_ctr / fps)
            print('time: ' + str(time))

            sums_diff_b = np.linalg.norm(sum_b - prev_emd_b)
            sums_diff_r = np.linalg.norm(sum_r - prev_emd_r)
            sums_diff_g = np.linalg.norm(sum_g - prev_emd_g)

            # max_sum = max([sum_b, sum_g, sum_r])
            max_sum_diff = max([sums_diff_b, sums_diff_r, sums_diff_g])
            print(max_sum_diff)
            threshes.append(max_sum_diff)

            previous_frame_b = processed_frame_b
            previous_frame_g = processed_frame_g
            previous_frame_r = processed_frame_r

            prev_emd_b = sum_b
            prev_emd_r = sum_r
            prev_emd_g = sum_g
            sampled = 0

        frame_ctr += 1
        sampled += 1

    hist, bin_edges = np.histogram(threshes, bins=5)
    threshes = [i for i in threshes if i >= bin_edges[2]]
    thresh = np.median(threshes)
    print(thresh)
    return thresh

# os.chdir('shots_ted_black_holes')
# os.chdir('SONS')
# train_sbd('shot_3.mp4')
# train_sbd('deep_learning_stanford.mp4')








