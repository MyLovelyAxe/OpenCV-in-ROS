#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# rostopic info /camera/rgb/image_raw:

# Type: sensor_msgs/Image
# Publishers:
#  * /gazebo (http://2_simulation:39257/)
# Subscribers: None

# rosmsg info sensor_msgs/Image:

# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# uint32 height
# uint32 width
# string encoding
# uint8 is_bigendian
# uint32 step
# uint8[] data


class Sobela_Class():

    def __init__(self):

        # define subscriber to topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()
        self.cv_bridge = CvBridge()

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # sobel operation along with horizontal direction
        self.sobel_hor = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        # sobel operation along with vertical direction
        self.sobel_ver = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)

        # show images
        cv2.imshow('original image', self.img)
        cv2.imshow('sobel_hor', self.sobel_hor)
        cv2.imshow('sobel_ver', self.sobel_ver)

        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("sobela_2_node")
    sobela = Sobela_Class()
    sobela.main()
