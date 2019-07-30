import cv2
import numpy as np

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


class Shot:
    def __init__(self, t, shot_title):
        self.starting_time = t
        self.ending_time = 0
        self.frames_arr = list()
        self.shot_stability = 0
        self.face_score = 0
        self.shot_len = 0
        self.title = shot_title

    def add_frame(self, img):
        self.frames_arr.append(img)

    def shot_stability_calc(self):
        shot_size = len(self.frames_arr)
        f_s = 0
        for k in range(1, shot_size):
            curr = self.frames_arr[k]
            curr = curr.astype(np.uint8)
            if k == 1:
                prev = np.zeros(curr.shape)
            else:
                prev = self.frames_arr[k-1]
            prev = prev.astype(np.uint8)
            f_s += edge_based_difference(prev, curr)
        f_s = f_s / (shot_size + 1)
        self.shot_stability = f_s
        # self.frames_arr.clear()
        return self.shot_stability

    def calculate_len(self):
        cap = cv2.VideoCapture(self.title)
        len = 0
        while True:
            ret, orig_frame = cap.read()
            if ret == False:
                break
            len += 1
        self.shot_len = len



