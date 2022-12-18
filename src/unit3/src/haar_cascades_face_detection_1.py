#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Haar_Cascades_Face_Detection_Class():

    def __init__(self):

        # get image
        self.scr_path = '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris.jpg'
        self.img = cv2.imread(self.scr_path)
        self.img = cv2.resize(self.img, (500, 300))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # create detectors
        self.face_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/frontalface.xml')
        self.eyes_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/eye.xml')

        # parameters for detection
        self.scale_factor = 1.2
        self.minNeighbors = 3

    def main(self):

        # refer to this link to get explanation of parameters of 'detectMultiScale':
        # https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
        self.res_face = self.face_detector.detectMultiScale(
            self.gray, self.scale_factor, self.minNeighbors)
        self.res_eyes = self.eyes_detector.detectMultiScale(
            self.gray, self.scale_factor, self.minNeighbors)

        # draw rectangles to illustrate face and eye positions on original image
        # result of 'detectMultiScale' is a 1x4 list with 4 element if only 1 object is detected:
        #   [[start_point_x, start_point_y, detected_area_width, detected_area_height]]
        # if n objects are detected, then the result is a nx4 list, similar with above
        for (x1, y1, w1, h1) in self.res_face:
            # cv2.rectangle(img_to_draw, start_point, end_point, color, thickness)
            # https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
            cv2.rectangle(self.img, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)
            # roi: area of interest
            # cut it out
            # attention: array index represents position as: [y,x]
            self.roi_face = self.img[y1:y1+h1, x1:x1+w1]

        for (x2, y2, w2, h2) in self.res_eyes:

            cv2.rectangle(self.img, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
            self.roi_face = self.img[y2:y2+h2, x2:x2+w2]

        cv2.imshow('detected_face_and_eyes', self.img)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("haar_cascades_face_detection_1_node")
    haar_cascades_face_detection = Haar_Cascades_Face_Detection_Class()
    haar_cascades_face_detection.main()
