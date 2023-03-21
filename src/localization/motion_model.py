import rospy
import numpy as np

class MotionModel:

    def __init__(self):
        self.deterministic = rospy.get_param("~deterministic")
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
        odom = np.array(odometry)
        particles_arr = np.array(particles)
        if not self.deterministic:
            # add Gaussian noise
            odom += np.random.normal(0, self.noise_st_dev, 3)
        particles_arr[:,0] = (np.cos(particles_arr[:,2]) * odom[0]
                              - np.sin(particles_arr[:,2]) * odom[1]
                              + particles_arr[:,0])
        particles_arr[:,1] = (np.sin(particles_arr[:,2]) * odom[0]
                              + np.cos(particles_arr[:,2]) * odom[1]
                              + particles_arr[:,1])
        particles_arr[:,2] += odom[2]
        return particles_arr.tolist()
