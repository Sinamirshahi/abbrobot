from libs.cot import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


cwd =os.getcwd() #getting curent woeking directory
lib = os.path.join(cwd,"libs")

prototxt_path = os.path.join(cwd,"libs","deploy.prototxt")
model_path = os.path.join(cwd,"libs","res10_300x300_ssd_iter_140000.caffemodel")
confidence_level = 0.4
simile = False


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# initialize the video stream and allow the camera sensor to warmup
print("tarting video stream...")
vs = VideoStream(src=0).start()

ids = [] # That's list of IDs of objects


# loop over the frames from the video stream
color = (0, 0, 255) #set the default colors which is red
att = False # later in code we can define if at this fram we have a center of attention or no
near = False # later in code we can define if the center of attention is close to the camera or no

if simile:
	smile_cascade = cv2.CascadeClassifier('libs/haarcascade_smile.xml')
 

while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		area = H * W
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
		if detections[0, 0, i, 2] > confidence_level:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")

			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0,0,255), 2)


			# size = (endX-startX) * (endY-startY)
			# cv2.putText(frame, str(size), (startX - 10, startY - 10),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	


	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	#find the oldest key so that would be the object which robot want to follow
	if len(list(objects.keys())) >= 1:
		attention = list(objects.keys())[0]

	
	# for item in list(objects.values()):
	# 	size_list.append((list(item))[2])
	# print(size_list)


	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame

		if attention == objectID: # check if the curent object is the one we locked our attention on
			color = (255,0,0)     #let it be blue

			att = True
			print ("Position: ",centroid," Object ID: ",objectID)

			# This part would be the coordinates X,Y we can tell the but to follow or whatever
			X = centroid[0] 
			Y = centroid[1]
			size = centroid[2]

			if size > int(area/20): #if size of the attention object is bigger than 20% (just an idea) of image 
				color = (0,255,0)
				near = True
				print("Near")      # any preferable action if the attention object was near
			else:
				pass
				# color = (0,0,255)



		text = "ID {}".format(objectID)

		#Here we put ID on the frame
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
		
		color = (0,0,255) # Reset the color for the next frame to processed as default red

		# cv2.putText(frame, str(centroid[2]), (centroid[0] + 10, centroid[1] + 10),  #putting size
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# cv2.putText(frame, str(size_list[counter]), (centroid[0] - 10, centroid[1] - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# # cv2.putText(frame, str(centroid[2]), (centroid[0] + 10, centroid[1] + 10),  #putting size
	if simile:
		smiles  = smile_cascade.detectMultiScale(frame, scaleFactor = 1.6, minNeighbors = 20)
		if smiles != ():
			print("smiled")
			for (sx, sy, sw, sh) in smiles:
				cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 255,0), 5)
 		


	# show the output frame
	cv2.imshow("Frame", frame)
	# cv2.imshow("flipped image", frame[:-1:])
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
