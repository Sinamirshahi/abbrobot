#!/usr/bin/env python3

from face_tracker_libs.cot import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from time import sleep


class FaceTracker(object):

    COL_FOUND = (0, 0, 255)
    COL_TRACKED = (255, 0, 0)
    COL_CLOSE = (0, 255, 0)

    def __init__(self, smile=False, conf_lvl=0.4, info=False, visual=False, flipped=True, near_thresh=0.2,
                 multipliers=(1, 1, 1), offsets=(0, 0, 0)):
        """
        :param bool smile: turn on smile detection
        :param float conf_lvl: confidence level
        :param bool info: print results (for debugging)
        :param bool visual: show visual representation of the tracking
        :param bool flipped: flip resulting image in visualization
        :param float near_thresh: threshold of the near detection
        :param tuple multipliers: (x,y,z) multipliers of the positions (pos = mult * pos + offs)
        :param tuple offsets: (x,y,z) offsets of the positions (pos = mult * pos + offs)

        get_face_pos        returns position of the face and whether smile was detected
        """

        self.proto_path = './face_tracker_libs/deploy.prototxt'
        self.model_path = './face_tracker_libs/res10_300x300_ssd_iter_140000.caffemodel'
        self.conf_lvl = conf_lvl  # confidence level
        self.info = info
        self.flipped = flipped
        self.visual = visual
        self.smile = smile
        self.near_thresh = near_thresh
        self.x = 50
        self.y = 50
        self.multipliers = multipliers
        self.offsets = offsets
        self.size = 0  # size can be transformed to z coordinate
        if self.smile:
            self.smile_cascade = cv2.CascadeClassifier('./face_tracker_libs/haarcascade_smile.xml')
        else:
            self.smile_cascade = None

        # initialize centroid tracker and frame dimensions
        self._ct = CentroidTracker()
        (self._H, self._W) = (None, None)
        self.area = 0

        # load serialized model
        if self.info:
            print("Loading model...")
        self._net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

        # initialize the video stream and allow the camera sensor to warmup
        if self.info:
            print("Starting video stream...")
        self._vs = VideoStream(src=0).start()

        self.att = False  # the current frame has an attention object
        self.near = False  # attention object is close to the camera

    def get_face_pos(self):
        """
        Get face coordinates, modified by multipliers and offsets of the class.

        :return: tuple: position = multiplier * coordinate + offset
        """
        # read the next frame from the video stream and resize it
        frame = self._vs.read()
        frame = imutils.resize(frame, width=400)
        smile_detected = False

        # if the frame dimensions are None, grab them
        if self._W is None or self._H is None:
            (self._H, self._W) = frame.shape[:2]
            self.area = self._H * self._W

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (self._W, self._H), (104.0, 177.0, 123.0))
        self._net.setInput(blob)
        detections = self._net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > self.conf_lvl:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([self._W, self._H, self._W, self._H])
                rects.append(box.astype('int'))

                if self.visual:
                    # draw a bounding box surrounding the object so we can
                    # visualize it
                    (startX, startY, endX, endY) = box.astype('int')
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    # size = (endX-startX) * (endY-startY)
                    # cv2.putText(frame, str(size), (startX - 10, startY - 10),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # update the centroid tracker using the computed set of bounding
        # box rectangles
        objects = self._ct.update(rects)

        # find the oldest key - to set actively tracked object
        if len(list(objects.keys())) >= 1:
            attention = list(objects.keys())[0]
        else:
            attention = None

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame

            if attention == objectID:  # check if the current object is the one we locked our attention on
                color = self.COL_TRACKED

                self.att = True
                if self.info:
                    print("Position: ", centroid, " Object ID: ", objectID)

                # This part would be the coordinates X,Y we can tell the but to follow or whatever
                self.x = self.offsets[0] + centroid[0].item()*self.multipliers[0]
                self.y = self.offsets[1] + centroid[1].item()*self.multipliers[1]
                self.size = self.offsets[2] + centroid[2].item()*self.multipliers[2]

                if self.size > int(self.area / self.near_thresh):
                    color = self.COL_CLOSE
                    self.near = True
                    if self.info:
                        print("Near")  # any preferable action if the attention object was near
                else:
                    self.near = False
            else:
                self.att = False
                color = self.COL_FOUND

            if self.visual:
                text = f'ID {objectID}'
                # Put ID on the frame
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

        if self.smile:
            smiles = self.smile_cascade.detectMultiScale(frame, scaleFactor=1.6, minNeighbors=20)
            if smiles != ():
                smile_detected = True
                if self.info:
                    print("Smile detected")
                if self.visual:
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 5)

        if self.visual:
            # show the output frame
            if self.flipped:
                cv2.imshow("Tracking", np.fliplr(frame))
            else:
                cv2.imshow("Tracking", frame)
        cv2.waitKey(1)

        return self.x, self.y, self.size, smile_detected

    def __del__(self):
        cv2.destroyAllWindows()
        self._vs.stop()


if __name__ == '__main__':
    ft = FaceTracker(visual=True, multipliers=(-50/255, 50/155, 1), offsets=(100, 0, 0))
    while True:
        x, y, _, _ = ft.get_face_pos()
        print(f'{x}, {y}')
        sleep(0.02)
