#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Feature_Matching_Class():

    def __init__(self):

        # get image train image
        self.train_img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_4/Course_images/ROS.png')
        self.train_img = cv2.resize(self.train_img, (300, 400))
        self.train_img = cv2.cvtColor(self.train_img, cv2.COLOR_BGR2GRAY)

        # get image query image
        self.query_img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_4/Course_images/ROS2.jpg')
        self.query_img = cv2.resize(self.query_img, (600, 400))
        self.query_img = cv2.cvtColor(self.query_img, cv2.COLOR_BGR2GRAY)

        # ORB object
        # https://docs.opencv.org/4.6.0/d1/d89/tutorial_py_orb.html
        self.orb_detector = cv2.ORB_create(nfeatures=1000)

        # Brute_Force Matcher object
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def main(self):

        ###########################################
        ###### Get Keypoints and Descriptors ######
        ###########################################

        # results of self.orb_detector.detectAndCompute():
        # keypoins:
        #       type: list with length [keypoints_num]
        #             here it is [967], which means we have 967 keypoints
        #       element in this list:
        #             all information of each keypoint, in which the attribute '.pt' is coordinates, e.g.[234.0, 315.0]
        # descriptors:
        #       type: numpy.ndarray with shape of [keypoints_number, descriptor_dimension]
        #             here it is [967,32], which means we have 967 keypoings with 32-dim descriptor
        self.train_kp, self.train_des = self.orb_detector.detectAndCompute(
            self.train_img, None)

        self.query_kp, self.query_des = self.orb_detector.detectAndCompute(
            self.query_img, None)

        # # show some results
        # rospy.loginfo("type and shape of keypoints:")
        # rospy.loginfo(str(type(self.train_kp)))
        # rospy.loginfo(str(len(self.train_kp)))
        # rospy.loginfo("type and shape of descriptor:")
        # rospy.loginfo(str(type(self.train_des)))
        # rospy.loginfo(str(self.train_des.shape))
        # rospy.loginfo("self.tranin_kp[0] = " + str(self.train_kp[0]))
        # rospy.loginfo("self.tranin_kp[0].pt = " + str(self.train_kp[0].pt))

        # draw keypoints
        # cv2.drawKeypoints(image, keypoints, outImage, flags)
        # https://docs.opencv.org/4.6.0/d4/d5d/group__features2d__draw.html#ga2c2ede79cd5141534ae70a3fd9f324c8
        self.train_draw_kp = cv2.drawKeypoints(
            self.train_img, self.train_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints image of train", self.train_draw_kp)

        self.query_draw_kp = cv2.drawKeypoints(
            self.query_img, self.query_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints image of query", self.query_draw_kp)

        ######################
        ###### Matching ######
        ######################

        # structure of return result of BFMatcher.macht(): a list of 'DMatch' objects
        # 'DMatch' object has following attributes:
        # DMatch.distance - Distance between descriptors. The lower, the better it is.
        # DMatch.trainIdx - Index of the descriptor in train descriptors
        # DMatch.queryIdx - Index of the descriptor in query descriptors
        # DMatch.imgIdx - Index of the train image.
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

        # result of self.matcher.match():
        # matches:
        #       type: list with length [matched_pair_number]
        #             here it is [253], which means there are 253 pair matched keypoints from train_img and query_img
        #       element in this list:
        #             type: cv2.DMatch
        #             content: see above
        self.matches = self.matcher.match(self.train_des, self.query_des)

        # # show something
        # rospy.loginfo("the structure in BFMatcher.match(): ")
        # rospy.loginfo("the type and shape: ")
        # rospy.loginfo(str(type(self.matches)) + str(len(self.matches)))
        # rospy.loginfo("the content inside: e.g. self.matches[0] ")
        # rospy.loginfo(str(type(self.matches[0])) + str(self.matches[0]))

        # Sort them in the order of their distance
        # The matches with shorter distance are the ones we want
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        # only reserve the most 20 matching pairs
        self.good_match = self.matches[:100]

        # draw matching lines
        # cv2.drawMatches(img1, keypoints1, img2, keypoints2, metaches1to2, outImg, flags):
        # https://docs.opencv.org/4.6.0/d4/d5d/group__features2d__draw.html#ga2c2ede79cd5141534ae70a3fd9f324c8
        self.match_outimage = cv2.drawMatches(
            self.train_img, self.train_kp, self.query_img, self.query_kp, self.good_match, None, flags=2)
        cv2.imshow("matches", self.match_outimage)

        #######################
        ###### Detection ######
        #######################

        # Parse the feature points
        train_points = np.float32(
            [self.train_kp[m.queryIdx].pt for m in self.good_match]).reshape(-1, 1, 2)
        test_points = np.float32(
            [self.query_kp[m.trainIdx].pt for m in self.good_match]).reshape(-1, 1, 2)

        # Create a mask to catch the matching points
        # With the homography we are trying to find perspectives between two planes
        # Using the Non-deterministic RANSAC method
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
        M, mask = cv2.findHomography(
            train_points, test_points, cv2.RANSAC, 5.0)

        # Catch the width and height from the main image
        h, w = self.train_img.shape[:2]

        # Create a floating matrix for the new perspective
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                         [w-1, 0]]).reshape(-1, 1, 2)

        # Create the perspective in the result
        dst = cv2.perspectiveTransform(pts, M)

        # Draw the points of the new perspective in the result image (This is considered the bounding box)
        result = cv2.polylines(self.query_img, [np.int32(
            dst)], True, (50, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("detection", result)

        cv2.waitKey(0)

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("feature_matching_1_node")
    feature_matching = Feature_Matching_Class()
    feature_matching.main()
