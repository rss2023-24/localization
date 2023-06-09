from __future__ import division
import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale", 1.0)


        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.squash = 1.0/2.2

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        p_hit_table = np.zeros((self.table_width, self.table_width))
        other_p_table = np.zeros((self.table_width, self.table_width))
        for d in range(self.table_width):
            for z in range(self.table_width):
                p_hit_table[z, d] = self.p_hit(z, self.table_width-1, d, self.sigma_hit)
                other_p_table[z, d] = (self.alpha_short * self.p_short(z, d) 
                                        + self.alpha_max * self.p_max(z, self.table_width-1) 
                                        + self.alpha_rand * self.p_rand(z, self.table_width-1))
        col_sums = p_hit_table.sum(axis=0)
        p_hit_table /= col_sums # normalize p_hit across d values
        self.sensor_model_table = other_p_table + self.alpha_hit * p_hit_table
        col_sums = self.sensor_model_table.sum(axis=0)
        self.sensor_model_table /= col_sums # normalize full table along columns
        

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # Generate Scan Data
        scans = self.scan_sim.scan(particles)
        
        # Convert measurements to pixel space
        conversion_factor = float(self.map_resolution)*self.lidar_scale_to_map_scale

        pixels_scans = scans / conversion_factor
        pixels_observation = observation / conversion_factor

        clipped_scans = np.clip(pixels_scans, 0, self.table_width - 1)
        clipped_observation = np.clip(pixels_observation, 0, self.table_width - 1)

        scaled_scans = np.rint(clipped_scans).astype(np.uint16)
        scaled_observation = np.rint(clipped_observation).astype(np.uint16)

        # Compute probablities
        probabilities = np.prod(self.sensor_model_table[scaled_observation, scaled_scans], axis = 1)

        # Smooth probability curve
        probabilities = np.power(probabilities, 1.0/2.2)
        return probabilities
        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        self.map_resolution = map_msg.info.resolution

        print("Map initialized")

    def p_hit(self, z, z_max, d, sigma):
        p = 0
        if 0 <= z and z <= z_max:
            exp_term = - (z-d)**2 / (2 * sigma**2)
            p = (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(exp_term)
        return p

    def p_short(self, z, d):
        p = 0
        if 0 <= z and z <= d and d != 0:
            p = (2/d) * (1 - z / d)
        return p

    def p_max(self, z, z_max):
        return 1 if z == z_max else 0

    def p_rand(self, z, z_max):
        p = 0
        if 0 <= z and z <= z_max:
            p = 1./ z_max
        return  p
