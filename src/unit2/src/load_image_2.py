#! /usr/bin/env python
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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


class ROS_CV2_Image_Class():

    def __init__(self):

        # define subscriber of topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            '/camera/rgb/image_raw', Image, self.camera_callback)
        self.img = Image()

        # define cv_bridge instance
        self.cv_bridge = CvBridge()

        # other things
        self.rate = rospy.Rate(1)
        self.count = 1

    def camera_callback(self, msg):

        self.img = msg
        # get image from drone and save it
        rospy.loginfo('showing ' + str(self.count) +
                      'th image from the drone......')
        # convert camera data into array
        try:
            image_from_drone = self.cv_bridge.imgmsg_to_cv2(
                self.img, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow('the image from the drone', image_from_drone)
        self.rate.sleep()
        self.count += 1

        # cv2.waitKey(i)
        # when i == 0, it only shows one image all the time
        # when i >= 1, it changes imaged approximately every 1 second
        # keep switching images from the drone
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("load_image_2_node")
    ROS_cv2_image = ROS_CV2_Image_Class()
    ROS_cv2_image.main()
