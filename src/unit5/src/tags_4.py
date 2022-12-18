#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco

###################################################################
###### detect markers in image & extract the center of artag ######
###################################################################


class Tags_Class():

    def __init__(self):

        # get source image
        self.img = cv2.imread(
            '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
        self.img_h, self.img_w = self.img.shape[:2]
        self.img = cv2.resize(
            self.img, (int(self.img_w*0.7), int(self.img_h*0.7)))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # initialize the dictionary
        self.tags_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.params = aruco.DetectorParameters_create()

        # to store the coordinates of markers' centers
        self.centers = []

    def coordinate_order(self, centers):

        # get a list 'centers' with 4 points, and order them in anticlockwise
        # attention1 :
        #   'centers' must be 'int' to be feasible in cv2.circle() and cv2.drawContours()
        order_centers = np.zeros((4, 2), dtype="int")
        # for example, the original 'center' from aruco.detectMarkers() is 7, 1, 4, 10
        # attention2 :
        #   the contents in 'centers' are (x,y) coordinates of 4 points
        #   and the coordinate system is shown as below:
        #   (the graph below is simulated the real coordinates from image, it is not a representation of np.array!!!!!!)
        #   ########################################################
        #   #### --------------------x-axis-------------------> ####
        #   ####  |                                             ####
        #   ####  |  center_4(x4,y4)      center_10(x10,y10)    ####
        #   #### y-axis                                         ####
        #   ####  |  center_7(x7,y7)      center_1(x1,y1)       ####
        #   ####  V                                             ####
        #   ########################################################

        # attention3:
        #   the order of the returned 'ids' is like:
        #       center7 --> center1 --> center4 --> center10
        #   but we need a anticlockwise order like this:
        #       center4 --> center10 --> center1 --> center7
        #   so we can do it according to properties of these 4 position:
        #   assume that center7 and center center10 are directly on the 45` diagonal, let's say:
        #   ################################################################
        #   #### -----------------------x-axis------------------------> ####
        #   ####  |                                                     ####
        #   ####  |  center_4(x4=0,y4=0)      center_10(x10=2,y10=0)    ####
        #   #### y-axis                                                 ####
        #   ####  |  center_7(x7=0,y7=2)      center_1(x1=2,y1=2)       ####
        #   ####  V                                                     ####
        #   ################################################################

        # so obviously, when we add value xi and yi, center4 and center1 will be the lowest and largest
        sum = centers.sum(axis=1)
        order_centers[0] = centers[np.argmin(sum)]  # center4
        order_centers[2] = centers[np.argmax(sum)]  # center1
        # and when we substract value yi to xi, center7 and center10 will be the largest and lowest(negative)
        # attention4:
        #   np.diff() calculate substraction as: the larger index substracts smaller index
        #   we actually use right digit to substract left digit, i.e. yi - xi
        #   https://numpy.org/doc/stable/reference/generated/numpy.diff.html
        diff = np.diff(centers, axis=1)
        order_centers[1] = centers[np.argmin(diff)]  # center10
        order_centers[3] = centers[np.argmax(diff)]  # center7
        # this is what we really want
        return order_centers

    def main(self):

        ###### detect the corners and id's in the examples ######

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            self.gray, self.tags_dict, parameters=self.params)

        rospy.loginfo("ids: " + str(ids))
        ###### centers of markers ######

        # ids is a np.array of shape (4,1) containing the id of detected markers
        for i in range(len(ids)):

            # corners is a list containing 4 elements
            # each element is a np.array with shape of (1,4,2)
            # which contains 4 corner coordinates of one marker
            marker = corners[i]
            # marker now is of shape (1,4,2)
            # but we only need (x,y) coordinates, a.k.a the (4,2) part
            marker = marker[0]
            # marker is of shape (4,2) containing 4 (x,y) coordinates
            # we need the mean of this 4 coordinates to get the position of this marker's center
            marker_center = marker.mean(axis=0)
            rospy.loginfo("coordinate of marker's center is: " +
                          str(marker_center))
            # use solid circle to mark the markers' centers
            # cv2.circle(image, center_coordinates, radius, color, thickness)
            # attention:
            #   I don't know why, but 'center_coordinates' must be 'int' type, or it will show errors
            # https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
            cv2.circle(
                self.img, (int(marker_center[0]), int(marker_center[1])), 3, (255, 255, 0), -1)
            # store markers' centers
            self.centers.append((int(marker_center[0]), int(marker_center[1])))

        # connect markers' centere with polygon
        self.centers = np.array(self.centers)

        # cv.drawContours(image, contours, contourIdx, color, thickness)
        # attention:
        #   when contourIdx is negative, all contours will be drawed
        #   if 'thickness' is positive, it represents thickness of polygon's egde lines, if it is '-1', then polygon is solid
        #   I don't know why, but 'contours' must be in '[ ]', or it will show errors
        # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
        without_order_img = self.img.copy()
        cv2.drawContours(without_order_img, [
                         self.centers], -1, (150, 150, 0), -1)
        aruco.drawDetectedMarkers(without_order_img, corners, ids)
        cv2.imshow("detected markers without ordering", without_order_img)

        # due to the fact that:
        # the polygon to be drawn is sensitive to the order of 'censters' given
        # so for now the outcome without sorting is shown as above
        # after sorting, we can obtain this:
        self.centers = self.coordinate_order(self.centers)
        print("self.centers after ordering: ")
        print(self.centers)
        with_order_img = self.img.copy()
        cv2.drawContours(with_order_img, [self.centers], -1, (150, 150, 0), -1)
        aruco.drawDetectedMarkers(with_order_img, corners, ids)
        cv2.imshow("detected markers with ordering", with_order_img)

        ###### draw markers ######

        # First we need to detect the markers itself, so we can later work with the coordinates we have for each.
        # output = aruco.drawDetectedMarkers(self.img, corners, ids)
        # cv2.imshow("detected markers", output)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_4_node")
    tags = Tags_Class()
    tags.main()
