import rospy
import numpy as np

class MotionModel:

    def __init__(self):
        self.deterministic = rospy.get_param("~deterministic")
        self.num_particles = rospy.get_param("~num_particles")
        self.noise_st_dev = 0.3


    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        if self.deterministic:
            particles[:,0] = (np.cos(particles[:,2]) * odometry[0]
                                - np.sin(particles[:,2]) * odometry[1]
                                + particles[:,0])
            particles[:,1] = (np.sin(particles[:,2]) * odometry[0]
                                + np.cos(particles[:,2]) * odometry[1]
                                + particles[:,1])
            particles[:,2] += odometry[2]

        else:
            # Create guassian noise
            odom_matrix = np.tile(odometry, (self.num_particles,1))
            odom_matrix += np.random.normal(0.0,self.noise_st_dev, (self.num_particles,3))
            
            # Update particle positions
            particles[:,0] = np.multiply(np.cos(particles[:,2]),odom_matrix[:,0]) - np.multiply(np.sin(particles[:,2]), odom_matrix[:,1]) + particles[:,0]
            particles[:,1] = np.multiply(np.sin(particles[:,2]), odom_matrix[:,0]) + np.multiply(np.cos(particles[:,2]), odom_matrix[:,1]) + particles[:,1]
            particles[:,2] += odom_matrix[:,2]
        return particles
