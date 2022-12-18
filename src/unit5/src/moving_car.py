#! /usr/bin/env python
import rospy
import time
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# rostopic info /cmd_vel:

# Type: geometry_msgs/Twist
# Publishers: None
# Subscribers:
#  * /gazebo (http://4_simulation:45321/)

# rosmsg info geometry_msgs/Twist:

# geometry_msgs/Vector3 linear
#   float64 x
#   float64 y
#   float64 z
# geometry_msgs/Vector3 angular
#   float64 x
#   float64 y
#   float64

# rostopic info /scan:

# Type: sensor_msgs/LaserScan
# Publishers:
#  * /gazebo (http://2_simulation:45425/)
# Subscribers: None

# rosmsg info sensor_msgs/LaserScan:

# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# float32 angle_min
# float32 angle_max
# float32 angle_increment
# float32 time_increment
# float32 scan_time
# float32 range_min
# float32 range_max
# float32[] ranges
# float32[] intensities


class Moving_Car_Class():

    def __init__(self):

        # subscribe to laser topic
        self.laser_sub = rospy.Subscriber(
            "/scan", LaserScan, self.sub_callback)
        self.laser_data = LaserScan()

        # velocity commands
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.vel = Twist()

    def sub_callback(self, msg):

        # get laser data
        self.laser_data = msg

    def main(self):

        rospy.loginfo("approaching the wall......")

        # wait for the sensor to get laser data
        time.sleep(3)

        # find the direction with shortest distance, which is the direction to the wall
        wall_dir = np.argmin(self.laser_data.ranges)
        rospy.loginfo("the shortest direction: " + str(wall_dir))
        wall_dist = self.laser_data.ranges[wall_dir]
        rospy.loginfo("distance to the wall: " + str(wall_dist))

        # determin the desired direction

        desired_dir = 0

        # define velocity
        # laser_data.ranges has length of 720, dividing the whole round into 720 intervals
        # every 2 intervals represents 1 degree, 0.0087 radius
        rot_vel = 0.0087
        # the angular distance between current direction and the orthometric frontal line
        # if current direction is on left of frontal line, then turn right to compensate, vise versa
        # in this case, the direction of 1 degree is frontal line
        angle_dist = wall_dir - desired_dir
        if angle_dist >= 360:
            angle_dist = angle_dist - 720

        rospy.loginfo('the angular distance is: ' + str(angle_dist))

        while wall_dist >= 1.0:

            # rotate to the wall
            while abs(angle_dist) >= 10:

                # rotate to the wall
                self.vel.angular.z = angle_dist * rot_vel / 2
                self.pub.publish(self.vel)
                time.sleep(2)
                self.vel.angular.z = 0.0
                self.pub.publish(self.vel)
                # update laser data
                rospy.loginfo("laser data: " + str(self.laser_data.ranges))
                wall_dir = np.argmin(self.laser_data.ranges)
                rospy.loginfo("the shortest direction: " + str(wall_dir))
                wall_dist = self.laser_data.ranges[wall_dir]
                rospy.loginfo("distance to the wall: " + str(wall_dist))
                angle_dist = wall_dir - desired_dir
                rospy.loginfo('the angular distance is: ' + str(angle_dist))
                if angle_dist >= 360:
                    angle_dist = angle_dist - 720

            # move forward to the wall
            self.vel.linear.x = 0.1
            self.pub.publish(self.vel)
            time.sleep(1)
            self.vel.linear.x = 0.0
            self.pub.publish(self.vel)

            # update laser data
            wall_dir = np.argmin(self.laser_data.ranges)
            rospy.loginfo("the shortest direction: " + str(wall_dir))
            wall_dist = self.laser_data.ranges[wall_dir]
            rospy.loginfo("distance to the wall: " + str(wall_dist))
            angle_dist = wall_dir - desired_dir
            rospy.loginfo('the angular distance is: ' + str(angle_dist))
            if angle_dist >= 360:
                angle_dist = angle_dist - 720

        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.1
        self.pub.publish(self.vel)
        time.sleep(2)
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0
        self.pub.publish(self.vel)
        rospy.loginfo("I stopped......")

        rospy.spin()


if __name__ == "__main__":

    rospy.init_node("moving_car_node")
    moving_car = Moving_Car_Class()
    moving_car.main()
