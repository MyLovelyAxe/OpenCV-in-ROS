#! /usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Take_Picture_Class():

    def __init__(self):

        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        self.cv_bridge = CvBridge()

        self.export_path = '/home/user/catkin_ws/src/unit4/image/well.png'

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('well', self.img)
        cv2.imwrite(self.export_path, self.img)

        cv2.waitKey(0)

    def main(self):

        rospy.loginfo("I am taking a picture......")

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("take_picture_node")
    take_picture = Take_Picture_Class()
    take_picture.main()
