# -*- coding: utf-8 -*-
from imutils.object_detection import non_max_suppression
#from PIL import Image
import numpy as np
import imutils
import cv2
import pytesseract
import os
from matplotlib import pyplot as plt
LINE_LEN = 12
MAX_SAMPLED = 10
config = '-l eng --oem 1 --psm 3'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'


def calc_avg_score(words):
    total = 0
    for word in words:
        total += word.score
    if len(words) == 0:
        return 0
    else:
        return total / len(words)


class Word:
    def __init__(self, score, text):
        self.score = score
        self.text = text
        self.length = len(text)


class TextualData:
    def __init__(self, words):
        self.words = words
        self.words_num = len(words)
        self.avg_score = calc_avg_score(self.words)


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    min_confidence = 0.5

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


# loop over frames from the video stream
def detect(img):
    # grab the current frame, then handle if we are using a
    # resize the frame, maintaining the aspect ratio

    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    width = np.size(img,1)#320
    height = np.size(img,0)#320
    (W, H) = (None, None)
    (newW, newH) = (width, height)
    (rW, rH) = (None, None)

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    model = 'frozen_east_text_detection.pb'
    net = cv2.dnn.readNet(model)

    img = imutils.resize(img, width=1000)
    orig = img.copy()

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = img.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio
    img = cv2.resize(img, (newW, newH))

    # construct a blob from the frame and then perform a forward pass
    # of the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(img, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    print(scores[0][0].shape)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # print(confidences)
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the frame
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig


def num_of_words(img):
    # Run tesseract OCR on image
    text = pytesseract.image_to_string(img, config=config)
    # Print recognized text
    # print(text)
    res = len(text.split())
    return res


def collect_textual_data_for_frame(img):
    # img_data = pytesseract.image_to_data(Image.open('sample1.jpg'))
    img_data = pytesseract.image_to_data(img)
    img_data = img_data.splitlines()
    words = []
    scores = []
    words_obj = []

    for line in range(1, len(img_data)):
        data_line = img_data[line]
        # print(data_line)
        data_list = data_line.split()
        if len(data_list) == 12 and int(data_list[-2]) > 0.5:
            words.append(data_list[-1])
            scores.append(int(data_list[-2]))

    for i in range(1,len(words)):
        word = Word(int(scores[i])/100, words[i])
        words_obj.append(word)

    return TextualData(words_obj)
def longest_string_correspondence(words1,num1,words2,num2):
    fit=np.zeros(num1)
    for i in range(0,num1):
        for j in range(0,num2):
            if words1[i].text==words2[j].text:
                fit[i]=1
                end_index=min(num1-i,num2-j)
                for k in range(0,end_index):
                    if words1[i+k].text==words2[j+k].text:
                        fit[i]=fit[i]+1
    return np.argmax(fit),np.max(fit)
if __name__=='__main__': 
    path=os.path.join('.\slidePics', '14.png')
#    os.path.join('slidePics')
    img = cv2.imread(path)
    img_quantized=np.round(img/8)*8
    img_quantized=np.asarray(img_quantized, dtype="uint8" )
    data=collect_textual_data_for_frame(img_quantized)
    last_num=data.words_num
    last_words=data.words
    m,n=np.size(img_quantized,0),np.size(img_quantized,1)
    i=1
    while last_num>3:
#        print(i,':')
        for j in range(0,last_num):
            print(last_words[j].text, end = ' ')
#        print()
        
        
        img_blurred = cv2.blur(img_quantized, (int(i*5),int(i*5*n/m)))
        data=collect_textual_data_for_frame(img_blurred)
        plt.figure()
        plt.imshow(img_blurred)
        num=data.words_num
        words=data.words
        index,length=longest_string_correspondence(last_words,last_num,words,num)
#        print(i+1,':')
#        for j in range(index,int(index+length-1)):
#            print(last_words[j].text, end = ' ')
#        print()
        if length<3:
            break
        last_words=words
        last_num=num
        last_index=index
        last_length=length
        
        i=i+1
    title=''
    if i>1:
        for j in range(last_index,int(last_index+last_length-1)):
            print(last_words[j].text, end = ' ')
            title=title+' '+last_words[j].text
#        title
    else:
        for j in range(0,last_num):
            print(last_words[j].text, end = ' ')
            title=title+' '+last_words[j].text