#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco

#####################################################
###### vreate markers with built-in dictionary ######
#####################################################


class Tags_Class():

    def __init__(self):

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        self.size = 700
        self.export_path = '/home/user/catkin_ws/src/unit5/tags/'

    def main(self):

        for i in range(1, 5):

            name = "tag_" + str(i) + ".jpg"

            # aruco.drawMarker(dictionary, id, sidePixels):
            # dictionary: defined as (DICT_6X6_250)
            # id:         the id that will be assigned. In this case, we have a loop so we will create 4 markers with the ids 1,2,3 and 4, respectively
            # sidePixels: the size of the output image, in this case, it will be 700x700 pixels.
            img = aruco.drawMarker(self.tags_dict, i, self.size)

            cv2.imshow("tag {}".format(i), img)
            cv2.imwrite(self.export_path + name, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_1_node")
    tags = Tags_Class()
    tags.main()
