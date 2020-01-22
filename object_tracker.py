# REF
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

# USAGE
#python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import json
import imutils
import time
import cv2

import intereactive_detection
import detectors
import camera
import argument_parser

is_age_gender_detection = False
flip_code = None

def dump_json(_object_list):
	results = {}
	for _object in _object_list:
		results.setdefault(_object[0], []).append(_object[1:])
	fw = open('./output.json','w')
	encode_json_data = json.dump(results, fw)

def main():
	#TODO: update
	args = argument_parser.build_argparser().parse_args()

	devices = [args.device, args.device_age_gender]
    models = [args.model_face, args.model_age_gender]

	if "CPU" in devices and args.cpu_extension is None:
    print("\nPlease try to specify cpu extensions library path in demo's command line parameters using -l ""or --cpu_extension command line argument")
    sys.exit(1)

	detections = interactive_detection.Detections(
        devices, models, args.cpu_extension, args.plugin_dir,args.prob_threshold, args.prob_threshold_face)
	#models = detections.models

	# TODO: ここを書き直す
	camera = VideoCamera(args.input, detections, args.no_v4l)
	# このframeにはすでに描画処理されているものだな
	while True:
		frame = camera.get_frame(flip_code, is_age_gender_detection)
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			dump_json(object_list)
			break
	# if we are using a webcam, release the pointer
	if video is None:
		vs.stop()
		dump_json(object_list)

	# otherwise, release the file pointer
	else:
		vs.release()
		dump_json(object_list)

	# do a bit of cleanup
	cv2.destroyAllWindows()


if __name__=='__main__':
    main()
