#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Pose, Point, PoseArray
from std_msgs.msg import Header


from threading import Lock
import numpy as np

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")
        self.num_particles = rospy.get_param("~num_particles")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")

        # Tunable Parameters 
        # TODO Tune this value
        self.noise_st_dev = 1
        # Initialize the models
        self.motion_model = MotionModel(self.noise_st_dev)
        self.sensor_model = SensorModel()

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
        
        # Particle Visualization
        self.visualizer = rospy.Publisher("/particles", PoseArray, queue_size = 15)

        # Create lock
        self.particle_lock = Lock()

        # Initialize particles
        self.particles = np.zeros((self.num_particles, 3))
        self.particle_indices = np.arange(0, self.num_particles)


    def initialize_robot_pose(self, msg):

        # Extract clickd position
        pose = msg.pose.pose
        covariance_matrix = np.array(msg.pose.covariance)
        orientation = pose.orientation
        roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        mean_position = [pose.position.x, pose.position.y, yaw]

        # Generate particles around clicked position
        initial_x = np.random.normal(mean_position[0], self.noise_st_dev, self.num_particles)
        initial_y = np.random.normal(mean_position[1], self.noise_st_dev, self.num_particles)
        initial_theta = np.random.normal(mean_position[2], self.noise_st_dev, self.num_particles)
        self.particles = np.column_stack((initial_x,initial_y,initial_theta))



    def handle_odometry(self, msg):
        # Lock particles array
        self.particle_lock.acquire()

        # Parse Odometry
        twist = msg.twist.twist
        odom_dx = twist.linear.x
        odom_dy = twist.linear.y
        odom_dtheta = twist.angular.z

        # Update particle positions
        self.particles = self.motion_model.evaluate(self.particles, np.array([odom_dx, odom_dy, odom_dtheta]))

        # Update robot pose
        self.compute_robot_pose()

        # Free particles array
        self.particle_lock.release()

    def handle_scan(self, msg):
        # Lock particles array
        self.particle_lock.acquire()

        # Downsample laser scan
        laserscan = np.array(msg.ranges)
        if len(laserscan) > self.num_beams_per_particle:
            idx = np.round(np.linspace(0, len(laserscan) - 1, self.num_beams_per_particle, endpoint=True)).astype(int)
            laserscan = laserscan[idx]

        # Compute particle probabilities
        probabilities = self.sensor_model.evaluate(self.particles, laserscan)
        if probabilities is None:
            self.particle_lock.release()
            return
        probabilities /= np.sum(probabilities) # normalize so probabilities sum to 1

        # Resample Particles
        sampled_indices = np.random.choice(self.particle_indices, self.num_particles, p=probabilities)
        self.particles = self.particles[sampled_indices]

        # Update robot pose
        self.compute_robot_pose()

        # Free particles array
        self.particle_lock.release()

    def compute_robot_pose(self):
        #TODO Figure out if there is a better way to estimate pose
        # Average location of all particles
        average_x = np.average(self.particles[:, 0])
        average_y = np.average(self.particles[:, 1])
        average_theta = np.arctan2(np.sum(np.sin(self.particles[:, 2])), np.sum(np.cos(self.particles[:, 2]))) # Circular Mean

        # Create Odometry object
        robot_odom = Odometry()
        robot_odom.header = Header(frame_id='/map')
        robot_odom.pose.pose.position = Point(average_x, average_y, 0)
        robot_odom.pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,average_theta))
 
        # Publish robot pose
        self.odom_pub.publish(robot_odom)

        # Visualize particles if subscribed
        if self.visualizer.get_num_connections() > 0:
            particles_poses_array = []
            particles_poses = PoseArray()

            # Build poses
            for i in range(self.num_particles):
                new_pose = Pose()
                new_pose.position = Point(self.particles[i, 0], self.particles[i, 1], 0)
                new_pose.orientation = Quaternion(*quaternion_from_euler(0,0,self.particles[i, 2]))
                particles_poses_array.append(new_pose)

            # Publish possees
            particles_poses.poses = particles_poses_array
            self.visualizer.publish(particles_poses)




if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
