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


class Sobelb_Class():

    def __init__(self):

        # define subscriber of topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        # manully define convolutional kernel
        self.kernel_ver = np.array([[-1, 0, 1], [-2, -0, 2], [-1, 0, 1]])
        self.kernel_hor = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

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

        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("sobelb_2_node")
    sobelb = Sobelb_Class()
    sobelb.main()
