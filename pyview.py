#! /usr/bin/env python3

'''
Simple Photo Collage application
Author: Oxben <oxben@free.fr>

-*- coding: utf-8 -*-
'''

import getopt
import json
import logging
import os
import signal
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QBoxLayout, QVBoxLayout, QSpacerItem
from PyQt5.QtWidgets import QToolBar
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsView, QGraphicsScene
from PyQt5.QtWidgets import QOpenGLWidget

from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap, QColor

from PyQt5.QtCore import Qt, QRect, QRectF
from player import Player

RotOffset = 5.0
ScaleOffset = 0.05
SmallScaleOffset = 0.01
MaxZoom = 2.0
FrameRadius = 15.0
MaxFrameRadius = 60.0
FrameWidth = 10.0
CollageAspectRatio = (16.0 / 9.0)
CollageSize = QRectF(0, 0, 2048, 1280 * (1 / CollageAspectRatio))
LimitDrag = True
OutFileName = ''
FrameColor = Qt.white
FrameBgColor = QColor(216, 216, 216)
LastDirectory = None

OpenGLRender = False

filenames = []
app = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoFrameItem(QGraphicsItem):
    '''The frame around a photo'''
    def __init__(self, rect, video_title, parent=None):
        super(PhotoFrameItem, self).__init__(parent)
        self.rect = rect
        self.photo = None
        self.video_dir = os.path.dirname(video_title)
        # Set flags
        self.setFlags(self.flags() |
                      QGraphicsItem.ItemClipsChildrenToShape |
                      QGraphicsItem.ItemIsFocusable)
        self.setAcceptDrops(True)
        self.setAcceptHoverEvents(True)
        # self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.panning = False

    def setPhoto(self, photo, reset=True):
        '''Set PhotoItem associated to this frame'''
        self.photo = photo
        self.photo.setParentItem(self)

    def boundingRect(self):
        '''Return bouding rectangle'''
        return QRectF(self.rect)

    def paint(self, painter, option, widget=None):
        '''Paint widget'''
        pen = painter.pen()
        pen.setColor(FrameColor)
        pen.setWidth(FrameRadius)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawRoundedRect(self.rect.left(), self.rect.top(),
                                self.rect.width(), self.rect.height(),
                                FrameRadius, FrameRadius)

    def hoverEnterEvent(self, event):
        '''Handle mouse hover event'''
        # Request keyboard events
        self.setFocus()

    def hoverLeaveEvent(self, event):
        '''Handle mouse leave event'''
        self.setCursor(Qt.PointingHandCursor)
        self.clearFocus()

    def mousePressEvent(self, event):
        # Panning
        self.setCursor(Qt.PointingHandCursor)
        title = ''
        fpath = ''
        os.chdir(self.video_dir)
        for file in os.listdir(self.video_dir):
            if file.endswith(".mp4"):
                title = file
                fpath = os.path.abspath(file)
                break
        print(os.path.splitext(title)[0])
        self.player = Player(os.path.splitext(title)[0])
        print(fpath)
        self.player.loadVideoFile(fpath)
        self.player.show()


class PhotoItem(QGraphicsPixmapItem):
    '''A photo item'''
    def __init__(self, filename):
        self.filename = filename
        super(PhotoItem, self).__init__(QPixmap(self.filename), parent=None)
        self.dragStartPosition = None
        # Use bilinear filtering
        self.setTransformationMode(Qt.SmoothTransformation)
        # Set flags
        self.setFlags(self.flags() |
                      QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemStacksBehindParent)

    def setPhoto(self, filename):
        pixmap = QPixmap(filename)
        if pixmap.width() > 0:
            logger.debug('SetPhoto(): %d %d', pixmap.width(), pixmap.height())
            self.filename = filename
            super(PhotoItem, self).setPixmap(pixmap)


class AspectRatioWidget(QWidget):
    '''Widget that keeps the aspect ratio of child widget on resize'''
    def __init__(self, widget, aspectRatio):
        super(AspectRatioWidget, self).__init__()
        self.layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout.addItem(QSpacerItem(0, 0))
        self.layout.addWidget(widget)
        self.layout.addItem(QSpacerItem(0, 0))
        self.setAspectRatio(aspectRatio)

    def setAspectRatio(self, aspectRatio):
        self.aspectRatio = aspectRatio
        self.updateAspectRatio()

    def updateAspectRatio(self):
        newAspectRatio = self.size().width() / self.size().height()
        if newAspectRatio > self.aspectRatio:
            # Too wide
            self.layout.setDirection(QBoxLayout.LeftToRight)
            widgetStretch = self.height() * self.aspectRatio
            outerStretch = (self.width() - widgetStretch) / 2 + 0.5
        else:
            # Too tall
            self.layout.setDirection(QBoxLayout.TopToBottom)
            widgetStretch = self.width() * (1 / self.aspectRatio)
            outerStretch = (self.height() - widgetStretch) / 2 + 0.5

        self.layout.setStretch(0, outerStretch)
        self.layout.setStretch(1, widgetStretch)
        self.layout.setStretch(2, outerStretch)

    def resizeEvent(self, event):
        self.updateAspectRatio()


class ImageView(QGraphicsView):
    '''GraphicsView containing the scene'''
    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def heightForWidth(self, w):
        logger.debug('heightForWidth(%d)', w)
        return w

    def resizeEvent(self, event):
        self.fitInView(CollageSize, Qt.KeepAspectRatio)


class CollageScene(QGraphicsScene):
    '''Scene containing the frames and the photos'''
    def __init__(self):
        super(CollageScene, self).__init__()
        self.bgRect = None
        self._initBackground()

    def addPhoto(self, rect, filepath):
        logger.info('Add image: %s', filepath)
        frame = PhotoFrameItem(QRect(0, 0, rect.width(), rect.height()), filepath)
        frame.setPos(rect.x(), rect.y())
        photo = PhotoItem(filepath)
        frame.setPhoto(photo)
        # Add frame to scene
        self.addItem(frame)

    def clear(self):
        super(CollageScene, self).clear()
        self._initBackground()

    def _initBackground(self):
        '''Add rect to provide background for PhotoFrameItem's'''
        pen = QPen(FrameBgColor)
        brush = QBrush(FrameBgColor)
        self.bgRect = QRectF(FrameWidth/2, FrameWidth/2,
                             CollageSize.width() - FrameWidth, CollageSize.height() - FrameWidth)
        self.addRect(self.bgRect, pen, brush)

    def getPhotosPaths(self):
        '''Return list containing the paths of all the photos in the scene'''
        paths = []
        items = self.items(order=Qt.AscendingOrder)
        if items:
            for item in items:
                if isinstance(item, PhotoItem):
                    paths.append(item.filename)
        logger.debug("Current photos: %s", str(paths))
        return paths


class LoopIter:
    '''Infinite iterator: loop on list elements, wrapping to first element when last element is reached'''
    def __init__(self, l):
        self.i = 0
        self.l = l

    def __iter__(self):
        return self

    def __next__(self):
        item = self.l[self.i]
        self.i = (self.i + 1) % len(self.l)
        return item

    def next(self):
        return self.__next__()


class PyView(QApplication):
    '''PyView class'''

    def __init__(self, argv):
        '''Constructor. Parse args and build UI.'''
        super(PyView, self).__init__(argv)
        self.win = None
        self.scene = None
        self.gfxView = None
        self.layoutCombo = None
        self.appPath = os.path.abspath(os.path.dirname(argv[0]))
        x = int(len(filenames) / 3) + len(filenames) % 3
        y = int(len(filenames) / 3) + 3 - len(filenames) % 3
        self.currentLayout = ('createGridCollage', (x, y))
        # Init GUI
        self.initUI()
        self.win.show()

    def initUI(self):
        '''Init UI of the PyView application'''
        # The QWidget widget is the base class of all user interface objects in PyQt5.
        self.win = QWidget()

        # Set window title
        self.win.setWindowTitle("PyView")
        self.win.resize(1000, 1000 * (1 / CollageAspectRatio))

        vbox = QVBoxLayout()
        self.win.setLayout(vbox)


        # Add toolbar
        toolbar = QToolBar()
        toolbar.setStyleSheet('QToolBar{spacing:5px;}')
        vbox.addWidget(toolbar)
        # Standard Qt Pixmaps: http://doc.qt.io/qt-5/qstyle.html#StandardPixmap-enum

        # Create GraphicsView
        self.gfxView = ImageView()
        self.arWidget = AspectRatioWidget(self.gfxView, CollageAspectRatio)
        vbox.addWidget(self.arWidget)

        # Set OpenGL renderer
        if OpenGLRender:
            self.gfxView.setViewport(QOpenGLWidget())

        # Add scene
        self.scene = CollageScene()

        # Create initial collage
        funcname, args = self.currentLayout
        self.setLayout(funcname, *args)

        self.gfxView.setScene(self.scene)

    def setLayout(self, funcname, *args):
        logger.debug('funcname=%s *args=%s', funcname, str(args))
        # Clear all items from scene
        self.scene.clear()
        # Create new collage
        func = getattr(self, funcname)
        if args:
            func(self.scene, *args)
        else:
            func(self.scene)

    def createGridCollage(self, scene, numx, numy):
        '''Create a collage with specified number of rows and columns'''
        if filenames:
            f = LoopIter(filenames)
            photoWidth = CollageSize.width() / numx
            photoHeight = CollageSize.height() / numy
            for x in range(0, numx):
                for y in range(0, numy):
                    scene.addPhoto(QRect(x * photoWidth, y * photoHeight, photoWidth, photoHeight), f.next())


def parse_args():
    '''Parse application arguments. Build list of filenames.'''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'Dh', ['help'])
    except getopt.GetoptError as err:
        logger.error(str(err))
        sys.exit(1)

    if args:
        for f in args:
            filenames.append(os.path.abspath(f))
            logger.debug(str(filenames))


def main():
    '''Main function'''
    global app
    parse_args()

    # Quit application on Ctrl+C
    # https://stackoverflow.com/questions/5160577/ctrl-c-doesnt-work-with-pyqt
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = PyView(sys.argv)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
