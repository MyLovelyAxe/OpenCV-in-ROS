#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Feature_Matching_Class():

    def __init__(self):

        # subscriber of camera topic
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.query_img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        # get image train image
        self.train_img = cv2.imread(
            '/home/user/catkin_ws/src/unit4/image/cropped_well.png')
        self.train_img = cv2.cvtColor(self.train_img, cv2.COLOR_BGR2GRAY)

        # ORB object
        self.orb_detector = cv2.ORB_create(nfeatures=1000)

        # Brute_Force Matcher object
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def camera_callback(self, msg):

        self.query_img = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

        ###########################################
        ###### Get Keypoints and Descriptors ######
        ###########################################

        self.train_kp, self.train_des = self.orb_detector.detectAndCompute(
            self.train_img, None)

        self.query_kp, self.query_des = self.orb_detector.detectAndCompute(
            self.query_img, None)

        # draw keypoints
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
        self.matches = self.matcher.match(self.train_des, self.query_des)

        # Sort them in the order of their distance
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        # only reserve the most 20 matching pairs
        self.good_match = self.matches[:100]

        # draw matching lines
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

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':

    rospy.init_node("feature_matching_2_node")
    feature_matching = Feature_Matching_Class()
    feature_matching.main()
