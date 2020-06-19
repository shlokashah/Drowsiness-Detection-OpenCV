from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

arg = argparse.ArgumentParser()
arg.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
arg.add_argument("-r", "--picamera", type=int, default=-1)
args = vars(arg.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow("Frame", frame)
	# plt.imshow(frame, interpolation = 'bicubic')
	# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	# plt.show()
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
