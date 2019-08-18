import sys
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
import time
from PyQt5 import QtCore, QtWidgets, uic
from player import Player


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Lecture Video App'
        self.initUI()
        self.download_dialog = DownloadWindow()
        self.player = None

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
        fileName, _ = QFileDialog.getOpenFileNames(self, "Open Files")
        title = os.path.basename(fileName[0])
        title = os.path.splitext(title)[0]
        self.player = Player(title)
        self.player.loadVideoFile(fileName[0])
        self.player.show()


class DownloadWindow(QtWidgets.QMainWindow):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DownloadWindow, self).__init__(parent)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.video_url = "https://www.youtube.com/watch?v=8S0FDjFBj8o&t=14s"
        self.le_url = QtWidgets.QLineEdit(self.video_url)
        path = os.getcwd()

        self.le_output = QtWidgets.QLineEdit(path)
        self.btn_download = QtWidgets.QPushButton("Download")
        self.progressbar = QtWidgets.QProgressBar(maximum=100)

        self.btn_download.clicked.connect(self.download)
        self.progressChanged.connect(self.progressbar.setValue)
        self.finished.connect(self.on_finished)

        form_lay = QtWidgets.QFormLayout(central_widget)
        form_lay.addRow("Url: ", self.le_url)
        form_lay.addRow("Output: ", self.le_output)
        form_lay.addRow(self.btn_download)
        form_lay.addRow(self.progressbar)

    @QtCore.pyqtSlot()
    def on_finished(self):
        self.update_disables(False)

    @QtCore.pyqtSlot()
    def download(self):
        self.pafy_video = pafy.new(self.video_url)
        self.video_id = self.pafy_video.videoid
        video_save = self.le_output.text()
        video_path = self.video_id
        try:
            os.makedirs(video_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        video_save = os.path.join(video_save, video_path)
        os.chdir(video_path)

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
        print(wholeText)
        with open(self.video_id + '.txt', "w") as text_file:
            text_file.write(wholeText)
        best = self.pafy_video.getbest(preftype='wvm')
        download_thread = threading.Thread(target=best.download, kwargs={'filepath': video_save, 'callback': self.callback}, daemon=True)
        download_thread.start()
        download_thread.join()

    def callback(self, total, recvd, ratio, rate, eta):
        val = int(ratio * 100)
        print(val)
        self.progressChanged.emit(val)
        if val == 100:
            self.finished.emit()
            time.sleep(0.5)
            print('Done')

    def update_disables(self, state):
        self.le_output.setDisabled(state)
        self.le_url.setDisabled(state)
        self.btn_download.setDisabled(not state)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.resize(320, 480)
    w.show()
    sys.exit(app.exec_())
