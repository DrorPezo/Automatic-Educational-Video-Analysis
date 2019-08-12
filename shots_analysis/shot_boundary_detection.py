import numpy as np
import cv2
import os
import errno
import skvideo.io
from moviepy.editor import VideoFileClip
from utils import Shot
from train_sbd import hists_dist


def video_shot(frames, file_name):
    frames = np.expand_dims(frames, axis=-1)
    output_data = np.asarray(frames)
    output_data = output_data.astype(np.uint8)
    skvideo.io.vwrite(file_name, output_data)


def sbd(video_title, rect_points, t):
    # these values should be selected by outliers detections by histogram of values
    cap = cv2.VideoCapture(video_title)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vt = os.path.splitext(video_title)[0]
    path = 'shots_' + vt
    first_frame = 1
    # 10 for short video with many animations. For longer video 150 is good
    MAX_SAMPLED = 10
    # MAX_SAMPLED = 150
    sampled = 0

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    curr_shot = Shot(0, ' ')
    frame_ctr = 0
    counter = 0
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

            max_sum = max([sum_b, sum_g, sum_r])
            max_sum_diff = max([sums_diff_b, sums_diff_r, sums_diff_g])
            print(max_sum_diff)

            if max_sum + max_sum_diff > t:
                if time - curr_shot.starting_time > 1.5:
                    print("Transition has been detected at " + "t = " + str(time) + " seconds")
                    counter += 1
                    curr_shot.ending_time = time
                    starting_time = curr_shot.starting_time
                    ending_time = curr_shot.ending_time
                    print('starting time: ' + str(curr_shot.starting_time) +
                          ' ending time: ' + str(curr_shot.ending_time))
                    print('starting time: ' + str(starting_time) + ' ending time: ' + str(ending_time))
                    clip = VideoFileClip(video_title).subclip(starting_time, ending_time)
                    shot_path = os.path.join(path, 'shot_' + str(counter) + '.mp4')
                    clip.write_videofile(shot_path)
                    clip.close()
                    print('--------Shot ' + str(counter) + ' has been saved------------')
                    curr_shot = Shot(time, ' ')

            previous_frame_b = processed_frame_b
            previous_frame_g = processed_frame_g
            previous_frame_r = processed_frame_r

            prev_emd_b = sum_b
            prev_emd_r = sum_r
            prev_emd_g = sum_g
            sampled = 0

        frame_ctr += 1
        sampled += 1

    # For the last shot
    counter += 1
    curr_shot.ending_time = float(frame_ctr / fps)
    starting_time = curr_shot.starting_time
    ending_time = curr_shot.ending_time
    print('starting time: ' + str(starting_time) + ' ending time: ' + str(ending_time))
    clip = VideoFileClip(video_title).subclip(starting_time, ending_time)
    shot_path = os.path.join(path, 'shot_' + str(counter) + '.mp4')
    clip.write_videofile(shot_path)
    clip.close()
    print('--------Shot ' + str(counter) + ' has been saved------------')
    print(str(counter) + " shots has been detected")
    print('--------Finish------------')


