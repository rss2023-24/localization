#!/usr/bin/env python2

import rospy
import tf2_ros
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Pose, Point, TransformStamped, PoseArray
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
        self.noise_st_dev = 0.7

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Create lock
        self.particle_lock = Lock()

        # Initialize variables
        self.particles = np.zeros((self.num_particles, 3))
        self.particle_indices = np.arange(0, self.num_particles)
        self.prev_time = rospy.get_time()

        # Initialize publishers/subscribers
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.handle_scan,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.handle_odometry, 
                                          queue_size=1) 
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                            self.initialize_robot_pose,
                                            queue_size=1)
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.transform_pub = tf2_ros.TransformBroadcaster()
        self.visualizer = rospy.Publisher("/particles", PoseArray, queue_size = 15)

        self.start_time = rospy.Time.now()
        self.update_times = np.array([])
        self.update_step_count = 0
        self.update_steps = np.array([])



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

        # Parse Odometry
        twist = msg.twist.twist
        odom_dx = twist.linear.x
        odom_dy = twist.linear.y
        odom_dtheta = twist.angular.z
        odom = np.array([odom_dx, odom_dy, odom_dtheta])

        # Scale odomotery by time passed
        curr_time = rospy.get_time() 
        odom *= (curr_time - self.prev_time)
        self.prev_time = curr_time

        # Update particle positions
        self.particle_lock.acquire()
        self.particles = self.motion_model.evaluate(self.particles, odom)

        # Update robot pose
        self.compute_robot_pose()

        # Free particles array
        self.particle_lock.release()

    def handle_scan(self, msg):

        # Downsample laser scan
        laserscan = np.array(msg.ranges)
        if len(laserscan) > self.num_beams_per_particle:
            idx = np.round(np.linspace(0, len(laserscan) - 1, self.num_beams_per_particle, endpoint=True)).astype(int)
            laserscan = laserscan[idx]

        # Compute particle probabilities
        self.particle_lock.acquire()
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

        curr_time = rospy.Time.now()

        average_x = np.average(self.particles[:, 0])
        average_y = np.average(self.particles[:, 1])
        average_theta = np.arctan2(np.sum(np.sin(self.particles[:, 2])), np.sum(np.cos(self.particles[:, 2]))) # Circular Mean

        # standard deviation of the center of mass 
        distances = np.sqrt(np.power(self.particles[:, 0] - average_x, 2) +  np.power(self.particles[:, 1] - average_y, 2))
        avg_distance = np.mean(distances)
        if (avg_distance > 0.1):
            self.update_step_count += 1
        else:
            self.update_steps = np.append(self.update_steps, self.update_step_count)
            self.update_step_count = 0

            time_diff = (curr_time - self.start_time).to_nsec() # In nanoseconds
            self.update_times = np.append(self.update_times, time_diff)
            self.start_time = curr_time

            print("Average update steps: {}".format(np.mean(self.update_steps)))
            print("Average update time: {}".format(np.mean(self.update_times)))

        # Create Transform
        transform_obj = TransformStamped()
        
        transform_obj.header = Header(frame_id='/map', stamp=curr_time)
        transform_obj.child_frame_id = self.particle_filter_frame
        transform_obj.transform.translation = Point(average_x, average_y, 0)
        transform_obj.transform.rotation = Quaternion(*quaternion_from_euler(0,0,average_theta))
        self.transform_pub.sendTransform(transform_obj)

        # Create Odometry object if subscribed
        if self.odom_pub.get_num_connections() > 0:
            robot_odom = Odometry()
            robot_odom.header = Header(frame_id='/map', stamp=curr_time)
            robot_odom.child_frame_id = self.particle_filter_frame
            robot_odom.pose.pose.position = Point(average_x, average_y, 0)
            robot_odom.pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,average_theta))
    
            # Publish robot pose
            self.odom_pub.publish(robot_odom)

        # Visualize particles if subscribed
        if self.visualizer.get_num_connections() > 0:
            particles_poses_array = []
            particles_poses = PoseArray()
            particles_poses.header = Header(frame_id='/map', stamp=curr_time)

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
