""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import numpy as np
import math
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from timeit import default_timer as timer

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

# FIXME: ここを標準入力でなんとかしたい
resize_prop = (640, 480)

class VideoCamera(object):
    def __init__(self, input, detections, no_v4l):
        if input == 'cam':
            self.input_stream = 0
            if no_v4l:
                self.cap = cv2.VideoCapture(self.input_stream)
            else:
                # for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
                try:
                    self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
                except:
                    import traceback
                    traceback.print_exc()
                    print(
                        "\nPlease try to start with command line parameters using --no_v4l\n"
                    )
                    os._exit(0)
        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        ret, self.frame = self.cap.read()
        cap_prop = self._get_cap_prop()
        logger.info("cap_pop:{}, frame_prop:{}".format(cap_prop, resize_prop))

        self.detections = detections

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, flip_code, is_age_gender_detection):

        ret, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, resize_prop)
        next_frame = None
        if self.input_stream == 0 and flip_code is not None:
            self.frame = cv2.flip(self.frame, int(flip_code))

        # face detectionの描画処理済みのframeが返ってくる
        frame = self.detections.face_detection(self.frame, next_frame, is_age_gender_detection)

        # TODO: 
        # ここにトラッキングの処理を書きたい


        # The first detected frame is None
        if frame is None:
            ret, jpeg = cv2.imencode('1.jpg', self.frame)
        else:
            ret, jpeg = cv2.imencode('1.jpg', frame)

        return jpeg.tostring()
