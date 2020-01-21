# REF
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import json
import imutils
import time
import cv2

def get_args():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	# 神宮のサンプルだと、condidence=0.2くらいが良さげ
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-v", "--video", type=str,
		help="path to input video file")
	args = ap.parse_args()
	return args


def dump_json(_object_list):
	results = {}
	for _object in _object_list:
		results.setdefault(_object[0], []).append(_object[1:])
	fw = open('./output.json','w')
	encode_json_data = json.dump(results, fw)

def main():
	args = get_args()
	model_1 = args.prototxt
	model_2 = args.model
	confidence = args.confidence
	video = args.video

	# initialize our centroid tracker and frame dimensions
	ct = CentroidTracker()
	(H, W) = (None, None)

	# load our serialized model from disk
	# TODO: change model to MOVIDIUS
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(model_1, model_2)

	# if a video path was not supplied, grab the reference to the web cam
	if video is None:
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(video)

	fps = None

	# objectのリストを格納する
	# TODO: キーをもたせたい
	object_list = []
	#object_list = { "objectID": objectID :{
				#"centroid_xy": centroid,
				#"frame_number" : frame_number,
		#}
		#}

	frame_number = 0

	# loop over the frames from the video stream
	while True:
		# read the next frame from the video stream and resize it
		frame = vs.read()
		frame = frame[1] if not video is None else frame
		# check to see if we have reached the end of the stream
		if frame is None:
			break

		frame = imutils.resize(frame, width=400)

		# if the frame dimensions are None, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the frame, pass it through the network,
		# obtain our output predictions, and initialize the list of
		# bounding box rectangles
		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
			(104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		rects = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# filter out weak detections by ensuring the predicted
			# probability is greater than a minimum threshold
			if detections[0, 0, i, 2] > confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the object, then update the bounding box rectangles list
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				rects.append(box.astype("int"))

				# draw a bounding box surrounding the object so we can
				# visualize it
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)

		# update our centroid tracker using the computed set of bounding
		# box rectangles
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			object_list.append((objectID, centroid.tolist(), frame_number))
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			dump_json(object_list)
			break

		frame_number += 1
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
