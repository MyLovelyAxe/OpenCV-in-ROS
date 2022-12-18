#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco

#####################################
###### detect markers in image ######
#####################################


class Tags_Class():

    def __init__(self):

        # get source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
        self.img_h, self.img_w = self.img.shape[:2]
        self.img = cv2.resize(
            self.img, (int(self.img_w*0.7), int(self.img_h*0.7)))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.params = aruco.DetectorParameters_create()

    def main(self):

        # Detect the corners and id's in the examples
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            self.img, self.tags_dict, parameters=self.params)

        # show something
        print("corners: ")
        # corners type: <class 'list'>
        # corners length: 4
        print(corners, type(corners), len(corners))
        print("corners[0]: ")
        # corners[0] type: <class 'numpy.ndarray'>
        # corners[0] shape: (1, 4, 2)
        print(corners[0], type(corners[0]), corners[0].shape)
        print("ids: ")
        # ids type: <class 'numpy.ndarray'>
        # ids shape: (4, 1)
        print(ids, type(ids), ids.shape)

        # First we need to detect the markers itself, so we can later work with the coordinates we have for each.
        output = aruco.drawDetectedMarkers(self.img, corners, ids)
        cv2.imshow("detected markers", output)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_2_node")
    tags = Tags_Class()
    tags.main()
