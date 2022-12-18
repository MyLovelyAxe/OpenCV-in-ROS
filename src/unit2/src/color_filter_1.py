#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# hsv: hue saturation value
# RGB-img is shown as BGR in python


class Color_Filter_Class():

    def __init__(self):

        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/Filtering.png')
        self.img = cv2.resize(self.img, (300, 300))
        self.hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def main(self):

        # define masks
        # these np.array is shown as: [Hue_limit, Saturation_limit, Value_limit]
        min_blue = np.array([110, 0, 0])
        max_blue = np.array([120, 255, 255])
        mask_blue = cv2.inRange(self.hsv_img, min_blue, max_blue)
        min_red = np.array([170, 0, 0])
        max_red = np.array([180, 255, 255])
        mask_red = cv2.inRange(self.hsv_img, min_red, max_red)
        min_green = np.array([50, 0, 0])
        max_green = np.array([60, 255, 255])
        mask_green = cv2.inRange(self.hsv_img, min_green, max_green)

        # apply masks
        result_blue = cv2.bitwise_and(self.img, self.img, mask=mask_blue)
        result_red = cv2.bitwise_and(self.img, self.img, mask=mask_red)
        result_green = cv2.bitwise_and(self.img, self.img, mask=mask_green)

        # show results
        cv2.imshow('image origin', self.img)
        cv2.imshow('image blue', result_blue)
        cv2.imshow('image red', result_red)
        cv2.imshow('image green', result_green)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("color_filter_1_node")
    color_filter = Color_Filter_Class()
    color_filter.main()
