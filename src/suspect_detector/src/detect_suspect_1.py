#! /usr/bin/env python
import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Detect_Suspect_Class():

    def __init__(self):

        ###### prepare for robot ######

        # subscribe to camera topic
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.camera_img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        ###### prepare for people detection ######

        # HOG detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

        ###### parepare for feature matching ######

        # get suspect wanted-post: wanted_img
        self.wanted_img = cv2.imread(
            "/home/user/catkin_ws/src/suspect_detector/images/wanted.png")
        self.wanted_gray = cv2.cvtColor(self.wanted_img, cv2.COLOR_BGR2GRAY)

        # ORB object
        # to get positions and desciptors of detection
        self.orb_detector = cv2.ORB_create(nfeatures=1000)

        # Brute Force Matcher
        # to match detection and train_image
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.best_match_number = 200

    def camera_callback(self, msg):

        #######################
        ###### get image ######
        #######################

        self.camera_img = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8")
        self.camera_gray = cv2.cvtColor(self.camera_img, cv2.COLOR_BGR2GRAY)

        ################################
        ####### detect all people ######
        ################################

        # get detected posisition
        # res_boxes: a list containing detected object with imformation of:
        #   x: starting point x
        #   y: starting point y
        #   w: detected region width
        #   h: detected region height
        # res_weights: the confidence of detections
        res_boxes, res_weights = self.hog_detector.detectMultiScale(
            self.camera_gray, winStride=(8, 8))

        # mark detected people
        detected_people_show = self.camera_img.copy()
        for (x, y, w, h) in res_boxes:
            cv2.rectangle(detected_people_show, (x, y),
                          (x+w, y+h), (255, 255, 0), 3)

        cv2.imshow("detected people", detected_people_show)

        ################################
        ###### feature extraction ######
        ################################

        # get descriptor of suspect
        wanted_kp, wanted_des = self.orb_detector.detectAndCompute(
            self.wanted_gray, None)
        wanted_kp_show = self.wanted_img.copy()
        wanted_kp_show = cv2.drawKeypoints(
            wanted_kp_show, wanted_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints of wanted", wanted_kp_show)

        # get descriptor of pedestrian
        people_kp, people_des = self.orb_detector.detectAndCompute(
            self.camera_gray, None)
        people_kp_show = self.camera_img.copy()
        people_kp_show = cv2.drawKeypoints(
            people_kp_show, people_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints of pedestrian", people_kp_show)

        ######################
        ###### matching ######
        ######################

        # find all match pairs without considering whether it is from the target
        match_pair = self.matcher.match(wanted_des, people_des)
        good_matches = sorted(match_pair, key=lambda x: x.distance)[
            :self.best_match_number]

        #################################
        ###### filtrate the target ######
        #################################

        # kp_ownship_count: how many match pairs does one detected person have?
        kp_ownship_count = 0
        # kp_ownship_num: a list whose elements represent:
        #       for each detected person, how many match pairs do they have?
        #       e.g. 3 people are detected, and kp_ownship_num = [200,30,40] means:
        #       1st person has 200 match pairs, 2nd person has 30 match pairs, and so on
        kp_ownship_num = []
        # kp_ownship_set_each: for each detected person, which match pairs do they have?
        kp_ownship_set_each = []
        # kp_ownship_set_whole: a list whose elements are 'kp_ownship_set_each'
        kp_ownship_set_whole = []

        # statistic for each detected person from camera
        # find the one getting the most match pairs, which is target
        for (x, y, w, h) in res_boxes:
            # to record how many keypoints are in this detected person's box
            kp_ownship_count = 0
            kp_ownship_set_each = []
            for pair in good_matches:
                # find the coordinate of keypoint in camera image of each match pair
                # attention:
                #       in order to find the points in query descriptor list, attribut 'trainIdx' of DMatch should be used
                #       (don't know why yet......)
                pt_pose = people_kp[pair.trainIdx].pt
                # check which detected person owns this keypoint
                # a.k.a whether pose of this keypoint lie in the box
                if pt_pose[0] <= (x+w) and pt_pose[0] >= x and pt_pose[1] <= (y+h) and pt_pose[1] >= y:
                    # another one match pair is confirmed for this detected person
                    # record the quantity
                    kp_ownship_count += 1
                    # record the pair
                    kp_ownship_set_each.append(pair)
            kp_ownship_num.append(kp_ownship_count)
            kp_ownship_set_whole.append(kp_ownship_set_each)

        # filtrate the target with most match pair
        # if there is person detected
        if not kp_ownship_num == []:
            target_idx = np.argmax(np.array(kp_ownship_num))
            best_match = kp_ownship_set_whole[target_idx]
        # if no person is detected
        else:
            best_match = good_matches

        wanted_match_show = self.wanted_img.copy()
        people_match_show = self.camera_img.copy()
        match_output = cv2.drawMatches(
            wanted_match_show, wanted_kp, people_match_show, people_kp, best_match, None, flags=2)
        cv2.imshow("matched output", match_output)

        #######################
        ###### detection ######
        #######################

        # Parse the feature points
        train_points = np.float32(
            [wanted_kp[m.queryIdx].pt for m in best_match]).reshape(-1, 1, 2)
        test_points = np.float32(
            [people_kp[m.trainIdx].pt for m in best_match]).reshape(-1, 1, 2)

        # Create a mask to catch the matching points
        hom, mask = cv2.findHomography(
            train_points, test_points, cv2.RANSAC, 2.0)

        # Catch the width and height from the main image
        h, w = self.wanted_img.shape[:2]

        # Create a floating matrix for the new perspective
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                         [w-1, 0]]).reshape(-1, 1, 2)

        # Create the perspective in the result
        dst = cv2.perspectiveTransform(pts, hom)

        # Draw the points of the new perspective in the result image (This is considered the bounding box)
        final_detection_show = self.camera_img.copy()
        final_detection_show = cv2.polylines(final_detection_show, [np.int32(
            dst)], True, (50, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("detection", final_detection_show)

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("detect_suspect_1_node")
    suspect_detector = Detect_Suspect_Class()
    suspect_detector.main()
