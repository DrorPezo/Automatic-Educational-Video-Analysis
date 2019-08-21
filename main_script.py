import sys
import os.path
from shots_analysis.shot_boundary_detection import sbd
from train_sbd import train_sbd
from train_sbd import train_thresh
from shots_analysis.shot_classifier import classify_all_shots
from shots_analysis.nons_static_animation_slide_classifier import extract_slides
# from shots_analysis.nons_static_animation_slide_classifier import classify_static_animation
# from shots_analysis.save_sos_static import save_static
# from shots_analysis.sos_static_animation_classifier import classify_sos
# from shots_analysis.slide_region_detection_static import save_slide
# from shots_analysis.slide_region_detection_animation import save_animation
from shot_utils import is_video_file


if __name__ == '__main__':
    arguments = len(sys.argv) - 1
    if arguments < 1:
        print('Error. No file was not found.')
        exit(1)
    elif arguments > 1:
        print('Error. Too many arguments.')
        exit(1)
    video_title = sys.argv[1]
    if not os.path.isfile(video_title):
        print('File does not exist.')
        exit(1)
    if not is_video_file(video_title):
        print('The file is not a video type.')
        exit(1)

    rect_data = train_sbd(video_title)
    thresh = train_thresh(video_title, rect_data)
    sbd(video_title, rect_data, thresh)
    classify_all_shots(video_title, None)
    classify_all_shots(video_title, rect_data)
    extract_slides(video_title, rect_data)
    # classify_sos(video_title)
    # save_static(video_title)
    # save_slide()
    # save_animation()

