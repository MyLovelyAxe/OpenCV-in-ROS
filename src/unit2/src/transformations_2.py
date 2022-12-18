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


class Transformations_Class():

    def __init__(self):

        # defien subscriber to topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # defeine cv_bridge
        self.cv_bridge = CvBridge()

        # kernel
        self.kernel = np.ones((5, 5), np.uint8)

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = cv2.resize(self.img, (450, 350))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

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

        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("transformations_2_node")
    transform = Transformations_Class()
    transform.main()
