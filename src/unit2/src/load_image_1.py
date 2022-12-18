#! /usr/bin/env python
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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

        # define cv_bridge instance
        self.cv_bridge = CvBridge()

        # read a picture
        self.image_to_read_path = '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_image_1.jpg'
        # save a picture
        self.image_to_save_path = '/home/user/catkin_ws/src/unit2/image/drone_image.jpg'

    def camera_callback(self, msg):

        # get image from drone and save it
        self.image_from_drone = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8")
        rospy.loginfo('show an image from the drone......')
        cv2.imshow('the image from the drone', self.image_from_drone)
        # if no specific path is designated
        # cv2.imwrite(image_name, image_array) will save the image in the current executing path of programming
        current_path = os.getcwd()
        rospy.loginfo(
            'the executing paht of programming is: ' + str(current_path))
        rospy.loginfo('the expecting path of saving is: ' +
                      str(self.image_to_save_path))
        cv2.imwrite(self.image_to_save_path, self.image_from_drone)

        cv2.waitKey(0)

    def main(self):

        # show an image
        rospy.loginfo('show an image that is already existed......')
        self.read_image = cv2.imread(self.image_to_read_path)
        cv2.imshow('the image to be read', self.read_image)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("load_image_1_node")
    ROS_cv2_image = ROS_CV2_Image_Class()
    ROS_cv2_image.main()
