import numpy as np
import cv2
import os


def shot_desc_arr(video_title):
    cap = cv2.VideoCapture(video_title)
    ret, frame = cap.read()
    frame_area = frame.shape[0] * frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    areas = []
    for i in range(16, 240, 16):
        ret, thresh = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = None
        max_contour_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                max_contour_area = area
                max_contour = contour
        if max_contour is not None:
            if len(max_contour) > 4:
                if max_contour_area > 0.1 * frame_area:
                    rect = cv2.minAreaRect(max_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    h = np.linalg.norm(box[0] - box[1])
                    w = np.linalg.norm(box[2] - box[3])
                    rect_area = h*w
                    if rect_area < 2 * max_contour_area:
                        # img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                        x0 = min([box[0][0], box[2][0]])
                        x1 = max([box[0][0], box[2][0]])
                        y0 = min([box[0][1], box[1][1]])
                        y1 = max([box[0][1], box[1][1]])
                        slide = frame[y0:y1, x0:x1]
                        # cv2.imshow('slide', slide)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        cv2.imwrite('slide of ' + video_title + '_' + str(i/16) + '.jpg', slide)


os.chdir('shots')
arr = os.listdir()
for file in arr:
    shot_desc_arr(file)


