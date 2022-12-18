#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from skimage import exposure, feature
from sensor_msgs.msg import Image

class HOG_People_Detection_Class():

    def __init__(self):

        # get video
        self.video = cv2.VideoCapture('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris5-2.mp4')

        # set HOG detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

        # shutdownhook
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdowhook)

        self.rate = rospy.Rate(10)

    def shutdowhook(self):

        # works better than the rospy.is_shutdown()
        self.ctrl_c = True

    def main(self):

        while not self.ctrl_c:

            # get image
            ret, frame = self.video.read()
            rospy.loginfo('ret: ' + str(ret))

            if ret == True:

                img = frame
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # define 8x8 blocks in the winStride
                res_boxes, res_weights = self.hog_detector.detectMultiScale(
                    gray, winStride=(8, 8))

                for ((x, y, w, h), confidence) in zip(res_boxes, res_weights):
                    # cv2.rectangle(img_to_draw, start_point, end_point, color, thickness)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)
                    rospy.loginfo(
                        "the confidence of the detection is: " + str(confidence))
                cv2.imshow('detected people', img)

                # visualization of HOG features
                self.hog_H, self.hog_feature = feature.hog(gray, orientations=9, pixels_per_cell=(
                    8, 8), cells_per_block=(2, 2), visualize=True)
                self.hog_feature = exposure.rescale_intensity(
                    self.hog_feature, out_range=(0, 255))
                self.hog_feature = self.hog_feature.astype("uint8")
                cv2.imshow('hog features', self.hog_feature)

                self.rate.sleep()
                cv2.waitKey(1)
            
            else:

                rospy.loginfo('the video is over......')
                self.ctrl_c = True
            
        self.video.release()

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("HOG_people_detection_2_node")
    HOG_people_detection = HOG_People_Detection_Class()
    HOG_people_detection.main()
