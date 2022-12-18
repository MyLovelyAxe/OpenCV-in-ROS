#! /usr/bin/env python

########################################
###### detect markers with camera ######
########################################

import rospy
import cv2
import numpy as np
from cv2 import aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Tags_Class():

    def __init__(self):

        # subscriber to camera topic
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.params = aruco.DetectorParameters_create()

    def camera_callback(self, msg):

        # get image
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img_h, self.img_w = self.img.shape[:2]
        self.img = cv2.resize(
            self.img, (int(self.img_w*1.5), int(self.img_h*1.5)))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Detect the corners and id's in the examples
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            self.gray, self.tags_dict, parameters=self.params)

        # First we need to detect the markers itself, so we can later work with the coordinates we have for each.
        output = aruco.drawDetectedMarkers(self.img, corners, ids)
        cv2.imshow("detected markers", output)

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_3_node")
    tags = Tags_Class()
    tags.main()
