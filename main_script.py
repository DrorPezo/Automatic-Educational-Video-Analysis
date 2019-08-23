import sys
import os.path
from shots_analysis.shot_boundary_detection import sbd
from shots_analysis.train_sbd import train_sbd
from shots_analysis.train_sbd import train_thresh
from shots_analysis.shot_classifier import classify_all_shots
from shots_analysis.sons_classifier import classify_sons
from shots_analysis.slides_extractor import extract_slides
from shots_analysis.save_slide import save
from shots_analysis.save_pdf import save_pdf
from utils.shot_utils import is_video_file
from shots_analysis.find_shot_key_words import add_key_words


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
    classify_sons(video_title)
    extract_slides(video_title, rect_data)
    save(video_title)
    save_pdf(video_title)
    add_key_words(video_title)

