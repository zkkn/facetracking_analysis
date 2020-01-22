from argparse import ArgumentParser
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import detectors
import cv2
import math
import numpy as np
from timeit import default_timer as timer
from queue import Queue

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

FP32 = "extension/IR/FP32/"
FP16 = "extension/IR/FP16/"

model_fc_xml = "face-detection-retail-0004.xml"
model_ag_xml = "age-gender-recognition-retail-0013.xml"


class Detectors(object):

    def __init__(self, devices, cpu_extension, plugin_dir,
                 prob_threshold, prob_threshold_face):
        self.cpu_extension = cpu_extension
        self.device_ss, self.device_fc, self.device_ag = devices
        self.plugin_dir = plugin_dir
        self.prob_threshold = prob_threshold
        self.prob_threshold_face = prob_threshold_face
        self._define_models()
        self._load_detectors()

    def _define_models(self):

        # set devices and models
        fp_path = FP32 if self.device_fc == "CPU" else FP16
        self.model_fc = fp_path + model_fc_xml
        fp_path = FP32 if self.device_ag == "CPU" else FP16
        self.model_ag = fp_path + model_ag_xml

        self.models = [self.model_fc, self.model_ag]

    def _load_detectors(self):

        # Create face_detection class instance
        self.face_detectors = detectors.FaceDetection(
            self.device_fc, self.model_fc, self.cpu_extension, self.plugin_dir, self.prob_threshold_face)
        # Create face_analytics class instances
        self.age_gender_detectors = detectors.AgeGenderDetection(
            self.device_ag, self.model_ag, self.cpu_extension, self.plugin_dir, self.prob_threshold_face)


class Detections(Detectors):
    def __init__(self, devices, models, cpu_extension, plugin_dir,
                 prob_threshold, prob_threshold_face):
        super().__init__(devices, cpu_extension, plugin_dir,
                         prob_threshold, prob_threshold_face)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def face_detection(self, frame, next_frame, is_age_gender_detection, ct):

        # ----------- Start Face Detection ---------- #

        logger.debug("** face_detection start **")
        color = (0, 255, 0)
        det_time = 0
        det_time_fc = 0
        det_time_txt = ""

        frame_h, frame_w = frame.shape[:2]  # shape (h, w, c)
        is_face_analytics_enabled = True if is_age_gender_detection else False

        inf_start = timer()
        self.face_detectors.submit_req(frame, next_frame)
        ret = self.face_detectors.wait()
        faces = self.face_detectors.get_results()
        inf_end = timer()
        det_time = inf_end - inf_start
        det_time_fc = det_time

        face_count = faces.shape[2]
        det_time_txt = "face_cnt:{} face:{:.3f} ms ".format(face_count,det_time * 1000)

        # ----------- Start Face Analytics ---------- #

        face_id = 0
        face_w, face_h = 0, 0
        face_frame = None
        next_face_frame = None
        prev_box = None
        det_time_ag = 0

        # Run face analytics with async mode when detected faces count are lager than 1.
        is_face_async_mode = False

        face_q = Queue()
        for face in faces[0][0]:
            face_q.put(face)

        rects = []
        for face_id in range(face_count):
            face_id = 0
            age_gender = ""

            if not face_q.empty():
                face = face_q.get()

            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")

            rects.append(box.astype("int"))

            class_id = int(face[1])
            result = str(face_id) + " " + str(round(face[2] * 100, 1)) + '% '

            if xmin < 0 or ymin < 0:
                logger.info(
                    "Rapid motion returns negative value(xmin and ymin) which make face_frame None. xmin:{} xmax:{} ymin:{} ymax:{}".
                    format(xmin, xmax, ymin, ymax))
                return frame

            # Start face analytics
            # prev_box is previous box(faces), which is None at the first time
            # will be updated with prev face box in async mode
            if is_face_async_mode:
                next_face_frame = frame[ymin:ymax, xmin:xmax]
                # if next_face_frame is None:
                # return frame
                if prev_box is not None:
                    xmin, ymin, xmax, ymax = prev_box.astype("int")
            else:
                face_frame = frame[ymin:ymax, xmin:xmax]

            # Check face frame.
            # face_fame is None at the first time with async mode.
            if face_frame is not None:
                face_w, face_h = face_frame.shape[:2]
                # Resizing face_frame will be failed when witdh or height of the face_fame is 0 ex. (243, 0, 3)
                if face_w == 0 or face_h == 0:
                    logger.error(
                        "Unexpected shape of face frame. face_frame.shape:{} {}".
                        format(face_h, face_w))
                    return frame

            # ----------- Start Age/Gender detection ---------- #
            if is_age_gender_detection:
                logger.debug("*** age_gender_detection start ***")

                inf_start = timer()
                self.age_gender_detectors.submit_req(
                    face_frame, next_face_frame, is_face_async_mode)
                ret = self.age_gender_detectors.wait()
                age, gender = self.age_gender_detectors.get_results(
                    is_face_async_mode)
                age_gender = str(int(round(age))) + " " + gender + " "
                inf_end = timer()
                det_time = inf_end - inf_start

                det_time_ag += det_time
                logger.debug("age:{} gender:{}".format(age, gender))
                logger.debug("*** age_gender_detection end ***")

            face_id += 1

            if is_face_async_mode:
                face_frame = next_face_frame
                prev_box = box

            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin), color, -1)
            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin),
                          (255, 255, 255))
            # Draw box and label\class_id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
            if is_face_analytics_enabled:
                cv2.putText(frame, age_gender, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            else:
                cv2.putText(frame, result, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            logger.debug("face_id:{} confidence:{}%".format(
                face_id, round(face[2] * 100)))

        # tracker processing
        # ここでトラッカーに突っ込んでる
        objects = ct.update(rects)
        # object_listもループの外部でグローバルに持ちたい
        # loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			#object_list.append((objectID, centroid.tolist(), frame_number))
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        det_time = det_time_fc + det_time_ag + det_time_em + det_time_hp + det_time_lm
        det_time_txt = det_time_txt + "ag:{:.2f} ".format(det_time_ag * 1000)

        frame = self.draw_perf_stats(det_time, det_time_txt, frame)

        return frame


    def draw_perf_stats(self, det_time, det_time_txt, frame):

        # Draw FPS in top left corner
        fps = self.calc_fps()
        cv2.rectangle(frame, (frame.shape[1] - 50, 0), (frame.shape[1], 17),
                      (255, 255, 255), -1)
        cv2.putText(frame, fps, (frame.shape[1] - 50 + 3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw performance stats
        inf_time_message = "Total Inference time: {:.3f} ms for sync mode".format(det_time * 1000)
        cv2.putText(frame, inf_time_message, (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        if det_time_txt:
            inf_time_message_each = "Detection time: {}".format(det_time_txt)
            cv2.putText(frame, inf_time_message_each, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        return frame

    def calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1

        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

        return self.fps