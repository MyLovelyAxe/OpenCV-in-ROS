#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Sobela_Class():

    def __init__(self):

        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img_b.jpg')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def main(self):

        # sobel operation along with horizontal direction
        self.sobel_hor = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        # sobel operation along with vertical direction
        self.sobel_ver = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)

        # show images
        cv2.imshow('original image', self.img)
        cv2.imshow('sobel_hor', self.sobel_hor)
        cv2.imshow('sobel_ver', self.sobel_ver)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("sobela_1_node")
    sobela = Sobela_Class()
    sobela.main()
