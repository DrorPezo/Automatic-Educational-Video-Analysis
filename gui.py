import sys
from langdetect import detect
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import pyqtSlot
import numpy as np
import pafy
from youtube_transcript_api import YouTubeTranscriptApi
import os
import errno
import threading
import subprocess
import re
import glob
import time
from PyQt5 import QtCore, QtWidgets, uic
from shot_utils import get_thumbnail_from_video


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Lecture Video App'
        self.initUI()
        self.download_dialog = DownloadWindow()
        self.player = None
        self.video_title = ''

    def initUI(self):
        self.setWindowTitle(self.title)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        button1 = QPushButton('Select Video', self)
        button1.setToolTip('Select video from lectures database')
        button1.move(100, 70)
        button1.clicked.connect(self.on_click_select_video)
        button2 = QPushButton('Download new video', self)
        button2.setToolTip('Download new lecture from YouTube')
        button2.move(100, 140)
        button2.clicked.connect(self.on_click_download_video)
        self.show()

    @pyqtSlot()
    def on_click_select_video(self):
        print('Select video from lecture database')
        self.open()

    @pyqtSlot()
    def on_click_download_video(self):
        print('Download new lecture from youtube')
        self.download_dialog.show()

    @pyqtSlot()
    def open(self):
        files_list = self.list_files_by_ext('mp4')
        print(files_list)
        for file in files_list:
            get_thumbnail_from_video(file)

        thunbnails_list = self.list_files_by_ext('jpg')
        # print(thunbnails_list)
        arg_list = []
        for file in thunbnails_list:
            file += ' '
            arg_list.append(file)

        files = ''.join(str(e) for e in arg_list)
        script = 'python pyview.py ' + files
        os.system(script)
        # fileName, _ = QFileDialog.getOpenFileNames(self, "Open Files", "Lectures")
        # print(fileName)

    def list_files_by_ext(self,ext):
        files_list = []
        for root, directories, files in os.walk('Lectures'):
            directories[:] = [d for d in directories if d not in ['shots']]
            for filename in files:
                if filename.endswith(ext):
                    filepath = os.path.join(root, filename)
                    if filepath:
                        files_list.append(filepath)

        return files_list


class DownloadWindow(QtWidgets.QMainWindow):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DownloadWindow, self).__init__(parent)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.video_url = ''
        self.le_url = QLineEdit(self.video_url)
        path = os.getcwd()

        self.le_output = QLineEdit(os.path.join(path, 'Lectures'))
        self.btn_download = QtWidgets.QPushButton("Download")
        self.progressbar = QtWidgets.QProgressBar(maximum=100)
        self.btn_download.clicked.connect(self.download)
        self.progressChanged.connect(self.progressbar.setValue)
        self.btn_analyze_video = QtWidgets.QPushButton("Analyze video")
        self.btn_analyze_video.setDisabled(True)
        self.finished.connect(self.on_finished)
        self.btn_analyze_video.clicked.connect(self.analyze_video)

        form_lay = QtWidgets.QFormLayout(central_widget)
        form_lay.addRow("Url: ", self.le_url)
        form_lay.addRow("Output: ", self.le_output)
        form_lay.addRow(self.btn_download)
        form_lay.addRow(self.progressbar)
        form_lay.addRow(self.btn_analyze_video)

    @QtCore.pyqtSlot()
    def on_finished(self):
        self.update_disables(False)
        file = glob.glob('*.mp4')
        print(file)
        file_sub = re.sub(' ', '_', file[0])
        if os.path.isfile(file[0]):
            os.rename(file[0], file_sub)

    @QtCore.pyqtSlot()
    def download(self):
        try:
            self.video_url = self.le_url.text()
        except Exception as e:
            print(e)
        self.pafy_video = pafy.new(self.video_url)
        self.video_id = self.pafy_video.videoid
        self.video_title = self.pafy_video.title
        self.video_title = re.sub('[!@#$|:"*?/<>]', '', self.video_title)
        self.video_title = re.sub('[ ]', '_', self.video_title)
        video_save = self.le_output.text()
        video_path = self.video_title
        os.chdir('Lectures')
        try:
            os.makedirs(video_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        video_save = os.path.join(video_save, video_path)
        os.chdir(video_path)
        lang = detect(video_path)
        if lang == 'en':
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id, languages=['en'])
            linesNum = np.size(transcript, 0)
            wholeText = ''
            for i in range(0, linesNum):
                line = transcript[i]
                text = line["text"]
                wholeText = wholeText + text
                if i < linesNum - 1:
                    if line["start"] + 2.5 <= transcript[i + 1]["start"]:
                        if line["start"] + 3.5 > transcript[i + 1]["start"]:
                            wholeText = wholeText + ','
                        elif line["start"] + line["duration"] - 1 < transcript[i + 1]["start"]:
                            wholeText = wholeText + '.'
                wholeText = wholeText + ' '
            # print(wholeText)
            with open(self.video_title + '.txt', "w") as text_file:
                text_file.write(wholeText)
        best = self.pafy_video.getbest(preftype='mp4')
        download_thread = threading.Thread(target=best.download, kwargs={'filepath': video_save, 'callback': self.callback}, daemon=True)
        download_thread.start()
        # download_thread.join()


    def callback(self, total, recvd, ratio, rate, eta):
        val = int(ratio * 100)
        # print(val)
        self.progressChanged.emit(val)
        if val == 100:
            self.finished.emit()
            print('Done')
            self.btn_analyze_video.setDisabled(False)
            print(os.getcwd())

    def update_disables(self, state):
        self.le_output.setDisabled(state)
        self.le_url.setDisabled(state)
        self.btn_download.setDisabled(not state)

    def analyze_video(self):
        script = "python " + os.path.join('..', '..', 'main_script.py ') + file_sub
        subprocess.call(script, shell=True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.resize(320, 480)
    w.show()
    sys.exit(app.exec_())
