#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# hsv: hue saturation value
# RGB-img is shown as BGR in python

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


class Color_Filter_Class():

    def __init__(self):

        # define subscriber of topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # cv_bridge instance
        self.cv_bridge = CvBridge()

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.img = cv2.resize(self.img, (300, 300))
        self.hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # define masks
        # these np.array is shown as: [Hue_limit, Saturation_limit, Value_limit]
        min_blue = np.array([90, 0, 0])
        max_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(self.hsv_img, min_blue, max_blue)
        min_red = np.array([0, 0, 0])
        max_red = np.array([20, 255, 255])
        mask_red = cv2.inRange(self.hsv_img, min_red, max_red)
        min_green = np.array([40, 0, 0])
        max_green = np.array([80, 255, 255])
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

        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        rospy.loginfo('filtering the image from the drone......')
        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("color_filter_2_node")
    color_filter = Color_Filter_Class()
    color_filter.main()
