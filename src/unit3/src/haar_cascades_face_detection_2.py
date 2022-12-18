#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Haar_Cascades_Face_Detection_Class():

    def __init__(self):

        # get 1-face image
        self.one_face = '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris.jpg'
        self.one_face_img = cv2.imread(self.one_face)
        self.one_face_img = cv2.resize(self.one_face_img, (500, 300))
        self.one_face_gray = cv2.cvtColor(
            self.one_face_img, cv2.COLOR_BGR2GRAY)

        # get multi-face image
        # because in this picture, faces and eyes are relatively too small compare to detector scale
        # so we don't resize the original image, in case faces and eyes are even smaller to be detected
        # in method 'detectMultiScale()' of 'cv2.CascadeClassifier'
        # there is a parameter:
        #   minSize – Minimum possible object size. Objects smaller than that are ignored.
        # as a result, too small faces and eyes cannot be detected
        self.many_face = '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/many.jpg'
        self.many_face_img = cv2.imread(self.many_face)
        # self.many_face_img = cv2.resize(self.many_face_img, (500, 300))
        self.many_face_gray = cv2.cvtColor(
            self.many_face_img, cv2.COLOR_BGR2GRAY)

        # create detectors
        self.face_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/frontalface.xml')
        self.eyes_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/eye.xml')

        # parameters for detection
        self.scale_factor = 1.2
        self.minNeighbors = 3

    def detection(self, face_img):

        # refer to this link to get explanation of parameters of 'detectMultiScale':
        res_face = self.face_detector.detectMultiScale(
            face_img, self.scale_factor, self.minNeighbors)

        res_eyes = self.eyes_detector.detectMultiScale(
            face_img, self.scale_factor, self.minNeighbors)

        # draw rectangles to illustrate face positions on original image
        for (x1, y1, w1, h1) in res_face:

            cv2.rectangle(face_img, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)

        # draw rectangles to illustrateeye positions on original image
        for (x2, y2, w2, h2) in res_eyes:

            cv2.rectangle(face_img, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

        rospy.loginfo('I am detecting faces......')

        return face_img

    def main(self):

        # detect one face
        one_face = self.one_face_gray
        one_face_img = self.detection(one_face)
        cv2.imshow('1 face', one_face_img)

        # detect multiple face
        many_face = self.many_face_gray
        many_face_img = self.detection(many_face)
        cv2.imshow('many faces', many_face_img)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("haar_cascades_face_detection_2_node")
    haar_cascades_face_detection = Haar_Cascades_Face_Detection_Class()
    haar_cascades_face_detection.main()
