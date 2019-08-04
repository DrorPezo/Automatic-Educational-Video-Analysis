import cv2
import numpy as np

POSITIVE = 1
NEGATIVE = -1
UNKNOWN = 0


def edge_based_difference(img1, img2):
    tau = 0.1
    N = 8
    win_w = int(img1.shape[0]/N)
    win_h = int(img1.shape[1]/N)
    i = 0
    changes = []
    # Crop out the window and process
    for r in range(0, img1.shape[0], win_w):
        for c in range(0, img1.shape[1] - win_h, win_h):
            window1 = img1[r:r + win_w, c:c + win_h]
            edges1 = cv2.Canny(window1, 100, 200)
            window2 = img2[r:r + win_w, c:c + win_h]
            edges2 = cv2.Canny(window2, 100, 200)
            d = np.linalg.norm(edges2 - edges1)
            changes.append(d)
            if d > tau:
                i += 1
    return changes, i


class Shot:
    def __init__(self, t, shot_title):
        self.starting_time = t
        self.ending_time = 0
        self.frames_arr = list()
        self.shot_stability = 0
        self.shot_stability_total = 0
        self.face_score = 0
        self.consistency = 0
        self.shot_len = 0
        self.title = shot_title
        self.f_t = 0
        self.shot_type = UNKNOWN

    def add_frame(self, img):
        self.frames_arr.append(img)

    def shot_stability_calc(self):
        shot_size = len(self.frames_arr)
        f_s = 0
        frames_edges_changes = []
        for k in range(1, shot_size):
            curr = self.frames_arr[k]
            curr = curr.astype(np.uint8)
            if k == 1:
                prev = np.zeros(curr.shape)
            else:
                prev = self.frames_arr[k-1]
            prev = prev.astype(np.uint8)
            changes, changed_blocks = edge_based_difference(prev, curr)
            med = np.median(changes)
            for n in range(0, len(changes)):
                var = changes[n] - med
                if var < 0:
                    var = 0
                changes[n] = var
            length = len(changes)
            changes = changes[5:length-5]
            frames_edges_changes.append(np.average(changes))
            f_s += changed_blocks

         # edge frames can be outlier
        total = sum(frames_edges_changes)/1000 # measure how the frames were changed
        f_s = f_s / (shot_size + 1)
        # print("sum: " + str(total))
        shot_stability_total = f_s # measure how many frames were changed
        # print('shot stability: ' + str(shot_stability_total))
        if f_s != 0:
            # print('ratio: ' + str(total/f_s))
            self.consistency = total/shot_stability_total
        self.shot_stability = frames_edges_changes.count(0)
        self.shot_stability_total = shot_stability_total
        self.frames_arr.clear()
        return self.shot_stability

    def calculate_len(self):
        cap = cv2.VideoCapture(self.title)
        fps = cap.get(cv2.CAP_PROP_FPS)
        len = 0
        while True:
            ret, orig_frame = cap.read()
            if ret == False:
                break
            len += 1
        self.shot_len = len
        self.ending_time = len/fps
        return len

    def classify_shot(self, theta_l, theta_s, theta_f, theta_c, theta_ct, theta_t, prev_s):
        length = self.calculate_len()
        if length > theta_l:
            # if self.shot_stability < theta_s and self.face_score < theta_f and self.f_t > theta_t \
            #         or 4 * self.shot_stability < prev_s:
            #     self.shot_type = POSITIVE
            # else:
            #     self.shot_type = NEGATIVE
            if self.shot_stability > theta_s or self.shot_stability_total < theta_s or \
                    (self.consistency < theta_c and self.consistency * self.shot_stability_total < theta_ct):
                self.shot_type = POSITIVE
            else:
                self.shot_type = NEGATIVE
        else:
            self.shot_type = UNKNOWN





