#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Sobelb_Class():

    def __init__(self):

        # source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img_b.jpg')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # manully define convolutional kernel
        self.kernel_ver = np.array([[-1, 0, 1], [-2, -0, 2], [-1, 0, 1]])
        self.kernel_hor = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def main(self):

        # apply kernels
        # image = cv2.filter2D(src, ddepth, kernel)
        # src: The source image on which to apply the fitler. It is a matrix that represents the image in pixel intensity values.
        # ddepth: It is the desirable depth of destination image. Value -1 represents that the resulting image will have same depth as the source image.
        # kernel: kernel is the filter matrix applied on the image.
        self.res_hor = cv2.filter2D(self.img, -1, self.kernel_hor)
        self.res_ver = cv2.filter2D(self.img, -1, self.kernel_ver)

        cv2.imshow('original image', self.img)
        cv2.imshow('horizontal result', self.res_hor)
        cv2.imshow('vertical result', self.res_ver)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("sobelb_1_node")
    sobelb = Sobelb_Class()
    sobelb.main()
