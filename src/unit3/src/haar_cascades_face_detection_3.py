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


class Haar_Cascades_Face_Detection_Class():

    def __init__(self):

        # define subscriber of topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber(
            '/camera/rgb/image_raw', Image, self.camera_callback)

        # cv_bridge
        self.cv_bridge = CvBridge()

        # create detectors
        self.face_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/frontalface.xml')
        self.eyes_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/eye.xml')

        # parameters for detection
        self.scale_factor = 1.2
        self.minNeighbors = 3

        # other things
        # indicate whether a face or eyes are detected
        self.indicator = False

    def camera_callback(self, msg):

        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.img = cv2.resize(self.img, (350, 550))

        self.res_face = self.face_detector.detectMultiScale(
            self.img, self.scale_factor, self.minNeighbors)
        self.res_eyes = self.eyes_detector.detectMultiScale(
            self.img, self.scale_factor, self.minNeighbors)

        # if there do exist a face
        if np.all(self.res_face):
            self.indicator = True
            for (x1, y1, w1, h1) in self.res_face:
                cv2.rectangle(self.img, (x1, y1),
                              (x1+w1, y1+h1), (255, 0, 255), 2)
        else:
            rospy.loginfo('no face has been detected for now......')

        if np.all(self.res_eyes):
            self.indicator = True
            for (x2, y2, w2, h2) in self.res_eyes:
                cv2.rectangle(self.img, (x2, y2),
                              (x2+w2, y2+h2), (0, 0, 255), 2)
        else:
            rospy.loginfo('no eye has been detected for now......')

        if self.indicator == True:
            cv2.imshow('detected face and eyes', self.img)
        else:
            cv2.imshow('nothing detected', self.img)
        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("haar_cascades_face_detection_3_node")
    haar_cascades_face_detection = Haar_Cascades_Face_Detection_Class()
    haar_cascades_face_detection.main()
