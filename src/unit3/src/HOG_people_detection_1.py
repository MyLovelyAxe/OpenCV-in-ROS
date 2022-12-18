#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from skimage import exposure, feature


class HOG_People_Detection_Class():

    def __init__(self):

        # image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/test_e.jpg')
        self.img = cv2.resize(self.img, (720, 1080))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # set HOG detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

    def main(self):

        # define 8x8 blocks in the winStride
        # the result of 'detectMultiScale()' of 'cv2.HOGDescriptor' has 2 parts:
        #   boxes:      similar with haar_cascade, which contains squares pointing to detected targets
        #   weights:    Vector that will contain confidence values for each detected object.
        self.res_boxes, self.res_weights = self.hog_detector.detectMultiScale(
            self.gray, winStride=(8, 8))

        # mark 3 person with different colors
        # attention:
        #   in cv2 the color encoding is BGR instead of RGB
        #   as a result, this 3 colors are: blue, green, red
        color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for idx, ((x, y, w, h), confidence, color) in enumerate(zip(self.res_boxes, self.res_weights, color_lst)):
            # cv2.rectangle(img_to_draw, start_point, end_point, color, thickness)
            cv2.rectangle(self.img, (x, y), (x+w, y+h), color, 2)
            rospy.loginfo(
                "the confidence of " + str(idx+1) + "th detection is: " + str(confidence))

        cv2.imshow('detected people', self.img)

        # visualization of HOG features
        self.hog_H, self.hog_feature = feature.hog(self.gray, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), visualize=True)
        self.hog_feature = exposure.rescale_intensity(
            self.hog_feature, out_range=(0, 255))
        self.hog_feature = self.hog_feature.astype("uint8")

        cv2.imshow('hog features', self.hog_feature)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("HOG_people_detection_1_node")
    HOG_people_detection = HOG_People_Detection_Class()
    HOG_people_detection.main()
