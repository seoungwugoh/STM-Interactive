import sys
from PyQt5.QtWidgets import (QWidget, QApplication, QMainWindow, QComboBox, QDialog,
        QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QSpinBox, QTextEdit, 
        QVBoxLayout, QMessageBox, QAction, QInputDialog, QColorDialog, QSizePolicy, QSlider, QLCDNumber, QSpinBox)

from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QImage, QPalette
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtCore import Qt
from PyQt5 import QtCore


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


import argparse
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import copy
import random
import glob

from model import model
from utils import load_frames_and_masks, load_frames, overlay_davis, overlay_checker, overlay_color, overlay_fade, overlay_cont
from jaccard import batched_f_measure, batched_jaccard

import qdarkstyle


SO_SET = ['blackswan', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'libby', 'parkour']


class App(QWidget):
    def __init__(self, batch_size):
        super().__init__()

        # buttons
        self.prev_button = QPushButton('Prev')
        self.prev_button.clicked.connect(self.on_prev)
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.on_next)
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.on_play)
        self.seg_button = QPushButton('Segment This')
        self.seg_button.clicked.connect(self.on_segment)
        self.run_button = QPushButton('Propagate to ALL')
        self.run_button.clicked.connect(self.on_run)
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.on_reset)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(100)
        

        # slide
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        
        
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slide)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("fade")
        self.combo.addItem("davis")
        self.combo.addItem("checker")
        self.combo.addItem("color")
        self.combo.currentTextChanged.connect(self.set_viz_mode)


        # combobox
        self.combo_seq = QComboBox(self)
        self.combo_seq.addItem('choose')
        seqs = glob.glob(os.path.join('sequences', '*'))
        for seq in seqs:
            if os.path.isdir(seq):
                seq_name = seq.split(os.sep)[-1]
                if seq_name not in ['choose'] and not seq_name.startswith('bg_layer') and not seq_name.startswith('mid_layer'):
                    self.combo_seq.addItem(seq.split(os.sep)[-1])
        self.combo_seq.currentTextChanged.connect(self.on_init)


        # # canvas
        # self.fig = plt.Figure()
        # self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        # self.ax.set_axis_off()
        # self.fig.add_axes(self.ax)
        # self.fig.set_facecolor((49/255.,54/255.,59/255. ))
        # self.canvas = FigureCanvas(self.fig)

        # self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        # self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.canvas = QLabel()
        self.canvas.setBackgroundRole(QPalette.Dark)
        # self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas.setScaledContents(True)

        self.canvas.mousePressEvent = self.on_press
        self.canvas.mouseReleaseEvent = self.on_release
        self.canvas.mouseMoveEvent = self.on_motion


        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addWidget(self.prev_button)
        navi.addWidget(self.play_button)
        navi.addWidget(self.next_button)
        navi.addStretch(1)
        navi.addWidget(QLabel('Overlay Mode'))
        navi.addWidget(self.combo)
        navi.addStretch(1)
        navi.addWidget(self.seg_button)
        navi.addWidget(self.run_button)
        navi.addWidget(self.combo_seq)
        navi.addWidget(self.reset_button)

        

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addLayout(navi)
        layout.setStretchFactor(navi, 1)
        layout.setStretchFactor(self.canvas, 0)
        self.setLayout(layout)

        # timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        self.re_init('libby')

    def re_init(self, sequence):
        self.this_round = -1

        self.sequence = sequence
        self.frames, self.masks = load_frames_and_masks('sequences/' + self.sequence)
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        # # init model
        self.model = model(self.frames, batch_size=batch_size)

        # set window
        self.setWindowTitle('Demo: STM interactive - Seq: {}'.format(self.sequence))
        self.setGeometry(100, 100, self.width*2, 2*self.height)

        self.lcd.setText('{: 3d} / {: 3d}'.format(0, self.num_frames-1))
        self.slider.setMaximum(self.num_frames-1)

        # initialize action
        self.reset_scribbles()
        self.pressed = False

        # initialize visualize
        self.viz_mode = 'fade'
        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        # self.slider.setValue(10)
        self.cursur = int(self.num_frames / 2.)
        self.on_showing = None
        self.show_current()

        self.start_time = time.time()
        self.show()
        # self.showFullScreen()
 
    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                    return qim.copy() if copy else qim


    def show_current(self):
        if self.viz_mode == 'fade':
            viz = overlay_fade(self.frames[self.cursur], self.current_mask[self.cursur]) 
        elif self.viz_mode == 'davis':
            viz = overlay_davis(self.frames[self.cursur], self.current_mask[self.cursur], rgb=[0, 0, 128]) 
            # viz = overlay_davis(self.frames[self.cursur], self.current_mask[self.cursur], rgb=[255, 0, 0]) 
        elif self.viz_mode == 'checker':
            viz = overlay_checker(self.frames[self.cursur], self.current_mask[self.cursur]) 
        elif self.viz_mode == 'color':
            viz = overlay_color(self.frames[self.cursur], self.current_mask[self.cursur], rgb=[223, 0, 223])
        elif self.viz_mode == 'cont':
            viz = overlay_cont(self.frames[self.cursur], self.current_mask[self.cursur]) 
        else:
            raise NotImplementedError

        self.current_pixmap = QPixmap.fromImage(self.toQImage(viz))
        self.canvas.setPixmap(self.current_pixmap)

        self.lcd.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames-1))
        self.slider.setValue(self.cursur)
        self.current_viz = viz

    def reset_scribbles(self):
        self.scribbles = {}
        self.scribbles['scribbles'] = [[] for _ in range(self.num_frames)]
        self.scribbles['sequence'] = self.sequence


    def set_viz_mode(self):
        self.viz_mode = self.combo.currentText()
        self.show_current()

    def slide(self):
        self.reset_scribbles()
        self.cursur = self.slider.value()
        self.show_current()
        # print('slide')

    def on_init(self):
        seq = self.combo_seq.currentText()
        self.sequence = seq
        self.re_init(self.sequence)

    def on_reset(self):
        self.re_init(self.sequence)

    def on_run(self):
        self.this_round += 1
        self.model.Segment('All')
        self.current_mask = self.model.Get_mask()
        # clear scribble and reset
        self.show_current()
        self.reset_scribbles()

        # evaluation
        gt_masks = self.masks
        pred_masks = self.current_mask
        
        if gt_masks is not None:
            Jscore = batched_jaccard(gt_masks, pred_masks)
            Fscore = batched_f_measure(gt_masks, pred_masks)
            ment = 'Round {} - {} | JF:[{:.3f}, {:.3f}]'.format(self.this_round+1, self.sequence, np.mean(Jscore), np.mean(Fscore))
        else:
            ment = 'Round {} - {} | JF: N/A'.format(self.this_round+1, self.sequence)
        print(ment)

        cv2.putText(self.current_viz, ment, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255])
        self.canvas.setPixmap(QPixmap.fromImage(self.toQImage(self.current_viz)))


    def on_segment(self):
        self.model.Memorize(self.scribbles)
        self.model.Segment(self.cursur)
        self.current_mask[self.cursur] = self.model.Get_mask_index(self.cursur)
        self.show_current()
        

    def on_prev(self):
        self.reset_scribbles()
        self.cursur = max(0, self.cursur-1)
        self.show_current()
        # print('prev')

    def on_next(self):
        self.reset_scribbles()
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.show_current()
        # print('next ')

    def on_time(self):
        self.reset_scribbles()
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.show_current()

    def on_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(1000 / 100)

    def on_press(self, event):
        x = event.pos().x() 
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width()) # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height()) # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))

        if x and y:
            # print('pressed', x, y)
            self.pressed = True
            self.stroke = {}
            self.stroke['path'] = []
            self.stroke['path'].append([norm_x, norm_y])
            if event.buttons() & Qt.LeftButton:
                self.stroke['object_id'] = 1
            else:
                self.stroke['object_id'] = 0
            self.stroke['start_time'] = time.time()
        self.draw_last_x = img_x
        self.draw_last_y = img_y


    def on_motion(self, event):
        x = event.pos().x() 
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width()) # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height()) # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))
        if self.pressed and x and y:
            # print('motion', x, y)
            self.stroke['path'].append([norm_x, norm_y])

            if self.stroke['object_id'] == 0:
                cv2.line(self.current_viz, (self.draw_last_x, self.draw_last_y), (img_x, img_y), color=[255,127,127], thickness=3)
            else:
                cv2.line(self.current_viz, (self.draw_last_x, self.draw_last_y), (img_x, img_y), color=[127,255,127], thickness=3)
            self.canvas.setPixmap(QPixmap.fromImage(self.toQImage(self.current_viz)))

            self.draw_last_x = img_x
            self.draw_last_y = img_y


    def on_release(self, event):
        x = event.pos().x() 
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width()) # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height()) # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))

        self.pressed = False
        if x and y:
            self.stroke['path'].append([norm_x, norm_y])
        
        self.stroke['end_time'] = time.time()
        self.scribbles['annotated_frame'] = self.cursur

        self.scribbles['scribbles'][self.cursur].append(self.stroke)
        
    


if __name__ == '__main__':
    
    def get_arguments():
        parser = argparse.ArgumentParser(description="args")
        parser.add_argument('-b','--batch-size', type=int, help="1,2,4,8 ...", required=False, default=1)
        return parser.parse_args()

    args = get_arguments()
    batch_size = args.batch_size

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App(batch_size)
    sys.exit(app.exec_())


