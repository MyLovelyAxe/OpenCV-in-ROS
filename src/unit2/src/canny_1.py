#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Canny_Class():

    def __init__(self):

        # source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img.png')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # manully define convolutional kernel
        self.min_value = 30
        self.max_value = 100

    def main(self):

        self.edge_img = cv2.Canny(self.img, self.min_value, self.max_value)

        cv2.imshow('original image', self.img)
        cv2.imshow('detected edges', self.edge_img)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("canny_1_node")
    canny = Canny_Class()
    canny.main()
