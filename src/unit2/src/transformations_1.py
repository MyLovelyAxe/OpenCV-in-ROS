#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Transformations_Class():

    def __init__(self):

        # source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/world.png')
        self.img = cv2.resize(self.img, (300, 300))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # kernel
        self.kernel = np.ones((5, 5), np.uint8)

    def main(self):

        # https://docs.opencv.org/4.6.0/d9/d61/tutorial_py_morphological_ops.html

        # erosion:
        # A pixel in the original image (either 1 or 0) will be considered 1
        # only if all the pixels under the kernel is 1,
        # otherwise it is eroded (made to zero).
        self.img_erosion = cv2.erode(self.img, self.kernel, iterations=1)

        # dilation:
        # a pixel element is '1' if at least one pixel under the kernel is '1'.
        # So it increases the white region in the image or size of foreground object increases
        self.img_dilation = cv2.dilate(self.img, self.kernel, iterations=1)

        # opening:
        # erosion followed by dilation
        # a.k.a firstly apply erosion, e.g. to discard scattered white noise points in background
        # then apply dilation to get cleaner image
        self.img_opening = cv2.morphologyEx(
            self.img, cv2.MORPH_OPEN, self.kernel)

        # closing:
        # dilation followed by erosion
        # a.k.a firstly apply erosion, e.g. to discard black noise points in foreground
        # then apply erosion to get cleaner image
        self.img_closing = cv2.morphologyEx(
            self.img, cv2.MORPH_CLOSE, self.kernel)

        cv2.imshow('original image', self.img)
        cv2.imshow('img_erosion', self.img_erosion)
        cv2.imshow('img_dilation', self.img_dilation)
        cv2.imshow('img_opening', self.img_opening)
        cv2.imshow('img_closing', self.img_closing)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("transformations_1_node")
    transform = Transformations_Class()
    transform.main()
