#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

###################################################
###### subtract artag area from camera image ######
###################################################


class Tags_Class():

    def __init__(self):

        # subscribe to camera topic
        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.img = Image()

        # cv_bridge
        self.cv_bridge = CvBridge()

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.params = aruco.DetectorParameters_create()

    def coordinate_order(self, centers, module):

        # get a list 'centers' with 4 points, and order them in anticlockwise
        order_centers = np.zeros((4, 2), dtype="int")

        # In the previous code, we sorted the coordinates to a specific order.
        # For this part, we need this order and also a second sort changing the position 2 and 3 of the array.
        # This is done just for the algorithm to work well.
        # When we work with the convexPoly, it works with a different order than the warped image with the homography.

        if module == 1:

            sum = centers.sum(axis=1)
            order_centers[0] = centers[np.argmin(sum)]
            order_centers[3] = centers[np.argmax(sum)]

            diff = np.diff(centers, axis=1)
            order_centers[1] = centers[np.argmin(diff)]
            order_centers[2] = centers[np.argmax(diff)]

        elif module == 2:

            sum = centers.sum(axis=1)
            order_centers[0] = centers[np.argmin(sum)]
            order_centers[2] = centers[np.argmax(sum)]

            diff = np.diff(centers, axis=1)
            order_centers[1] = centers[np.argmin(diff)]
            order_centers[3] = centers[np.argmax(diff)]

        return order_centers

    def camera_callback(self, msg):

        # get image
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_h, self.img_w = self.img.shape[:2]

        ###### detect the corners and id's in the examples ######

        # to store the coordinates of markers' centers
        self.centers = []

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            self.gray, self.tags_dict, parameters=self.params)
        rospy.loginfo("ids: " + str(ids))

        ###### centers of markers ######

        for i in range(len(ids)):
            # draw centers of markers with circle
            marker = corners[i]
            marker = marker[0]
            marker_center = marker.mean(axis=0)
            rospy.loginfo("coordinate of marker's center is: " +
                          str(marker_center))
            cv2.circle(
                self.img, (int(marker_center[0]), int(marker_center[1])), 3, (255, 255, 0), -1)
            self.centers.append((int(marker_center[0]), int(marker_center[1])))
        # order the centers
        self.centers = np.array(self.centers)
        self.centers = self.coordinate_order(self.centers, module=2)
        print("self.centers after ordering: ")
        print(self.centers)
        # draw contours
        ordered_img = self.img.copy()
        cv2.drawContours(ordered_img, [self.centers], -1, (150, 150, 0), -1)
        aruco.drawDetectedMarkers(ordered_img, corners, ids)
        cv2.imshow("detected markers with ordering", ordered_img)

        ###### create black mask ######

        black_mask = np.zeros([self.img_h, self.img_w, 3], dtype=np.uint8)
        cv2.fillConvexPoly(black_mask, np.int32(
            [self.centers]), (255, 255, 255), cv2.LINE_AA)
        cv2.imshow('black mask', black_mask)

        ###### subtraction ######

        subtraction = cv2.subtract(ordered_img, black_mask)
        cv2.imshow("subtraction", subtraction)

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.loginfo("getting image from camera and subtracting......")
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_7_node")
    tags = Tags_Class()
    tags.main()
