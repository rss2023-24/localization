#!/usr/bin/env python2

import numpy as np

import rospy
from ackermann_msgs.msg import AckermannDriveStamped

class TestDrive:
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT

    DRIVE_TOPIC = "/drive"
    VELOCITY = 0.75


    def __init__(self):
        self.drive_pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        self.drive()

    def drive(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            ts = AckermannDriveStamped()
            ts.drive.speed = self.VELOCITY
            self.drive_pub.publish(ts)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('test_drive')
    test_drive = TestDrive()
    rospy.spin()
