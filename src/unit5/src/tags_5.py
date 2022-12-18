#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from cv2 import aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

######################################################################
###### detect markers from camera & extract the center of artag ######
######################################################################


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

    def coordinate_order(self, centers):

        # get a list 'centers' with 4 points, and order them in anticlockwise
        order_centers = np.zeros((4, 2), dtype="int")

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

        # to store the coordinates of markers' centers
        self.centers = []

        ###### detect the corners and id's in the examples ######

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            self.gray, self.tags_dict, parameters=self.params)

        ###### centers of markers ######

        for i in range(len(ids)):

            marker = corners[i]
            marker = marker[0]
            marker_center = marker.mean(axis=0)
            # mark centers of markers with circles
            cv2.circle(
                self.img, (int(marker_center[0]), int(marker_center[1])), 3, (255, 255, 0), -1)
            self.centers.append((int(marker_center[0]), int(marker_center[1])))

        # order the coordinates
        self.centers = np.array(self.centers)
        self.centers = self.coordinate_order(self.centers)
        print("self.centers: ")
        print(self.centers)

        ###### draw contours ######

        with_order_img = self.img.copy()
        cv2.drawContours(with_order_img, [self.centers], -1, (150, 150, 0), -1)
        aruco.drawDetectedMarkers(with_order_img, corners, ids)
        cv2.imshow("detected markers with ordering", with_order_img)

        cv2.waitKey(1)

    def main(self):

        try:
            rospy.loginfo("processing images from camera......")
            rospy.spin()
        except CvBridgeError as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == "__main__":

    rospy.init_node("tags_5_node")
    tags = Tags_Class()
    tags.main()
