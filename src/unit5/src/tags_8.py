#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco

###################################################
###### subtract artag area from given image ######
###################################################


class Tags_Class():

    def __init__(self):

        # get source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
        self.img_h, self.img_w = self.img.shape[:2]
        self.img_h = int(self.img_h * 0.7)
        self.img_w = int(self.img_w * 0.7)
        self.img = cv2.resize(
            self.img, (self.img_w, self.img_h))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # get iamge to be added
        self.added_img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/earth.jpg')
        self.added_h, self.added_w = self.added_img.shape[:2]
        self.added_coordinates = np.array(
            [[0, 0], [self.added_w, 0], [0, self.added_h], [self.added_w, self.added_h]])

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.params = aruco.DetectorParameters_create()

        # to store the coordinates of markers' centers
        self.centers = []

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

    def main(self):

        ###### detect the corners and id's in the examples ######

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
        # cv2.imshow('black mask', black_mask)

        ###### subtraction ######

        subtraction = cv2.subtract(ordered_img, black_mask)
        cv2.imshow("subtraction", subtraction)

        ###### addtition of added image ######

        # get homography matrix from added image to the shape of polygon with 'centers' as corners
        hom, status = cv2.findHomography(self.added_coordinates, self.centers)
        # https://theailearner.com/tag/cv2-warpperspective/
        warped_img = cv2.warpPerspective(
            self.added_img, hom, (self.img_w, self.img_h))
        cv2.imshow("warped image", warped_img)

        addition = cv2.add(warped_img, subtraction)
        cv2.imshow("addition", addition)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_8_node")
    tags = Tags_Class()
    tags.main()
