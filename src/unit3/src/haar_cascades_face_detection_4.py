#! /usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Haar_Cascades_Face_Detection_Class():

    def __init__(self):

        # get video
        self.scr_path = '/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris.mp4'
        self.video = cv2.VideoCapture(self.scr_path)

        # create detectors
        self.face_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/frontalface.xml')
        self.eyes_detector = cv2.CascadeClassifier(
            '/home/user/catkin_ws/src/unit3/haar_cascades/eye.xml')

        # parameters for detection
        self.scale_factor = 1.2
        self.minNeighbors = 3

        # shutdown hook
        # https://get-help.robotigniteacademy.com/t/use-of-rospy-on-shutdown/2873
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

        self.rate = rospy.Rate(10)

    def shutdownhook(self):

        # works better than the rospy.is_shutdown()
        self.ctrl_c = True

    def main(self):

        while not self.ctrl_c:

            # https://stackoverflow.com/questions/13989627/cv2-videocapture-read-does-not-return-a-numpy-array
            # self.video.read() from a VideoCapture returns a tuple (return value, image).
            #   ret:    you check wether the reading is successful, if it is True, then you proceed to use the returned image.
            #   frame:  the image for processing
            ret, frame = self.video.read()
            rospy.loginfo('what does this ret show here: ' + str(ret))

            if ret == True:

                img_ori = cv2.resize(frame, (300, 200))
                img_dec = cv2.resize(frame, (300, 200))

                gray_dec = cv2.cvtColor(img_dec, cv2.COLOR_BGR2GRAY)

                # refer to this link to get explanation of parameters of 'detectMultiScale':
                res_face = self.face_detector.detectMultiScale(
                    gray_dec, self.scale_factor, self.minNeighbors)
                res_eyes = self.eyes_detector.detectMultiScale(
                    gray_dec, self.scale_factor, self.minNeighbors)

                # draw rectangles to illustrate face and eye positions on original image
                for (x1, y1, w1, h1) in res_face:
                    # cv2.rectangle(img_to_draw, start_point, end_point, color, thickness)
                    cv2.rectangle(img_ori, (x1, y1),
                                  (x1+w1, y1+h1), (255, 0, 255), 2)
                    # roi: area of interest
                    # self.roi_face = self.img[y1:y1+h1, x1:x1+w1]

                for (x2, y2, w2, h2) in res_eyes:

                    cv2.rectangle(img_ori, (x2, y2),
                                  (x2+w2, y2+h2), (0, 0, 255), 2)
                    # self.roi_face = self.img[y2:y2+h2, x2:x2+w2]

                cv2.imshow('detected_face_and_eyes', img_ori)
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

    rospy.init_node("haar_cascades_face_detection_4_node")
    haar_cascades_face_detection = Haar_Cascades_Face_Detection_Class()
    haar_cascades_face_detection.main()
