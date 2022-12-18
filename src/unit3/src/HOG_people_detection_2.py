#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from skimage import exposure, feature
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

class HOG_People_Detection_Class():

    def __init__(self):

        # define subscriber of topic '/camera/rgb/image_raw'
        self.camera_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.camera_callback)
        self.img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        # set HOG detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

    def camera_callback(self, msg):

        # get image
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img = cv2.resize(self.img, (700, 500))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # define 8x8 blocks in the winStride
        self.res_boxes, self.res_weights = self.hog_detector.detectMultiScale(
            self.gray, winStride=(8, 8))

        for ((x, y, w, h), confidence) in zip(self.res_boxes, self.res_weights):
            # cv2.rectangle(img_to_draw, start_point, end_point, color, thickness)
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (255,0,255), 2)
            rospy.loginfo(
                "the confidence of the detection is: " + str(confidence))
            # check whether the man is on the right or left
            if (x + 0.5 * w) <= 350:
                rospy.loginfo('the man is on the left')
            else:
                rospy.loginfo('the man is on the right')

        cv2.imshow('detected people', self.img)

        # visualization of HOG features
        self.hog_H, self.hog_feature = feature.hog(self.gray, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), visualize=True)
        self.hog_feature = exposure.rescale_intensity(
            self.hog_feature, out_range=(0, 255))
        self.hog_feature = self.hog_feature.astype("uint8")

        cv2.imshow('hog features', self.hog_feature)

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("HOG_people_detection_2_node")
    HOG_people_detection = HOG_People_Detection_Class()
    HOG_people_detection.main()
