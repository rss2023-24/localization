#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from threading import Lock
import numpy as np

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.handle_scan,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.handle_odometry, 
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initialize_robot_pose,
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Create lock
        self.particle_lock = Lock()

        # Initialize particles
        self.particles = np.zeros((self.num_particles, 3))
        self.particle_indices = np.arange(0, self.num_particles)

    def initialize_robot_pose(self, msg):
        return 

    def handle_odometry(self, msg):
        pass 

    def handle_scan(self, msg):
        # Lock particles array
        self.particle_lock.acquire()

        # Downsample laser scan
        raw_laserscan = np.array(msg.ranges)
        idx = np.round(np.linspace(0, len(raw_laserscan) - 1, self.num_beams_per_particle, endpoint=True)).astype(int)
        downsampled_laserscan = raw_laserscan.ranges[idx]

        # Compute particle probabilities
        self.sensor_model.evaluate(self.particles, downsampled_laserscan)

        # Resample Particles


        # Update robot pose
        self.compute_robot_pose()


        # Free particles array
        self.particle_lock.release()

    def compute_robot_pose(self):
        pass



if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
