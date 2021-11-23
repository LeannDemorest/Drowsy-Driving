# Based on PyimageSearch eye blink detection tutorial 
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/


from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import subprocess
import os

# make sure to install/upgrade imutils $ pip install --upgrade imutils

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())

# define a constant for the eye aspect ratio to indicate a blink 
EYE_AR_THRESH = 0.2
drop_threshold = 50
lag = 21 #delay before prompting user
init_period = 5 #moving average window for nose
# initialize the frame counters
COUNTER = 0
total_frames = 0
drop_counter = 0
drop_flag = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
datFile =  "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(datFile)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

""" # start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# uncomment below line for raspberry pi camera
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0) """

#toggle for recorded video
rec_video = True

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
#fileStream = True
if rec_video == False:
	vs = VideoStream(src=0).start()
# uncomment below line for raspberry pi camera
# vs = VideoStream(usePiCamera=True).start()
fileStream = False

if rec_video == True: 
    vs = cv2.VideoCapture("IMG_7475.MOV")
time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()
nose_tip = []
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	if rec_video == True:
		ret, frame = vs.read()
		if ret == False:
			break
	else:
		frame = vs.read()
	total_frames += 1
	
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		#get nose position
		nose = shape[nStart:nEnd]
		#l_eyebrow = shape[lbStart:lbEnd]
		#r_eyebrow = shape[rbStart:rbEnd]
		nose_tip.append(nose[6][1]) #append tip of nose to list
		#initialize nose when looking straight ahead
		if total_frames == init_period:
			mov_avg = np.convolve(nose_tip, np.ones(5), 'valid')/5
		if total_frames>= init_period:
			drop = nose_tip[-1]-mov_avg[-1] #compute head drop
			if drop > drop_threshold:
				drop_counter += 1
				#cv2.putText(frame, 'Look up', (10,frame.shape[0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			if drop_counter >=lag: #detect prolonged drop
					print('Lift your head!')
					#cv2.putText(frame, 'Look up', (10,frame.shape[0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
					if rec_video == False:
						subprocess.call(["espeak", "-s 5 -ven", "Look up"])
					drop_counter = 0
					drop_flag = 1    
			if drop < drop_threshold and drop_flag == 1: #detect looking up
				print("Good job looking up")
				drop_flag = 0
		cv2.circle(frame, (nose[6][0], nose[6][1]), radius = 4, color = (0, 255, 0), thickness = 1) #draw nose
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			#cv2.putText(frame, 'Wake up', (10,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			if COUNTER >= lag:
				print('Wake up!')
				#cv2.putText(frame, 'Wake up', (10,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				if rec_video == False:
					subprocess.call(["espeak", "-s 5 -ven", "You may have fallen asleep"])
				COUNTER = 0
 	
 	# update the FPS counter
	fps.update()

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
if rec_video == True:
	vs.release()
else:
	vs.stop()
cv2.destroyAllWindows()
