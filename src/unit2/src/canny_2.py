#! /usr/bin/env python
import rospy
import numpy as np
import cv2
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


class Canny_Class():

    def __init__(self):

        # defien subscriber to topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # defeine cv_bridge
        self.cv_bridge = CvBridge()

        # manully define convolutional kernel
        self.min_value = 30
        self.max_value = 100

    def camera_callback(self, msg):

        # convert original image
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # apply canny
        self.edge_img = cv2.Canny(self.img, self.min_value, self.max_value)

        cv2.imshow('original image', self.img)
        cv2.imshow('detected edges', self.edge_img)

        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("canny_2_node")
    canny = Canny_Class()
    canny.main()
