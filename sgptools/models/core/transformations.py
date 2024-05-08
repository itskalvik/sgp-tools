# Copyright 2024 The SGP-Tools Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides transforms to model complex sensor field of views and handle informative path planning
"""

try:
    from tensorflow_graphics.math.interpolation import bspline
except:
  pass

import tensorflow as tf
import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


class Transform:
    """Base class for transformations of the inducing points, including expansion and aggregation transforms.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]
        - Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [Jakkala and Akella, 2024]

    Args:
        aggregation_size (int): Number of consecutive inducing points to aggregate
        constraint_weight (float): Weight term that controls the importance of the 
                                   constraint terms in the SGP's optimization objective 
    """
    def __init__(self, 
                 aggregation_size=None, 
                 constraint_weight=1.0,
                 **kwargs):
        self.aggregation_size = aggregation_size
        self.constraint_weight = constraint_weight

    def expand(self, Xu):
        """Applies the expansion transform to the inducing points

        Args:
            Xu (ndarray): Expansion transformed inducing points
        """
        return Xu

    def aggregate(self, k):
        """Applies the aggregation transform to kernel matrices

        Args:
            k (tensor): (mp, mp)/(mp, n); Kernel matrix. 
                        `m` is the number of inducing points,
                        `p` is the number of points each inducing point is mapped,
                        `n` is the number of training data points.

        Returns:
            k (tensor): (m, m)/(m, n); Aggregated kernel matrix
        """
        if self.aggregation_size is None:
            return k

        if k.shape[0] == k.shape[1]:
            # Handle Kuu which is a square matrix
            k = tf.expand_dims(tf.expand_dims(k, axis=0), axis=-1)
            k = tf.nn.avg_pool(k,
                               ksize=[1, self.aggregation_size, self.aggregation_size, 1],
                               strides=[1, self.aggregation_size, self.aggregation_size, 1],
                               padding='VALID')
            k = tf.squeeze(k, axis=[0, -1])
        else:
            # Handle Kuf which is a rectangular matrix
            k = tf.expand_dims(k, axis=0)
            k = tf.nn.avg_pool(k,
                               ksize=[1, self.aggregation_size, 1],
                               strides=[1, self.aggregation_size, 1],
                               padding='VALID')
            k = tf.squeeze(k, axis=[0])
        return k

    def constraints(self, Xu):
        """Computes the constraint terms that are added to the SGP's optimization function

        Args:
            Xu (ndarray): Inducing points from which to compute the constraints

        Returns:
            c (float): constraint terms (eg., distance constraint)
        """
        return 0.

    def distance(self, Xu):
        """Computes the distance incured by sequentially visiting the inducing points

        Args:
            Xu (ndarray): Inducing points from which to compute the path length

        Returns:
            dist (float): path length
        """
        dist = tf.math.reduce_sum(tf.norm(Xu[1:]-Xu[:-1], axis=1))
        return dist
    

class SquareTransform(Transform):
    """Non-point Transform to model a square FoV. Only works for single robot cases. 
    ToDo: update expand function to handle multi-robot case.

    Args:
        length (float): Length of the square FoV
        num_side (int): Number of points along each side of the FoV
    """
    def __init__(self, length, num_side, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.num_side = num_side
        self.length_factor=length/(self.num_side)
        self.num_length = int(length/self.length_factor)

        if self.aggregation_size == 0:
            self.aggregation_size = None
        elif self.aggregation_size is None:
            self.aggregation_size = num_side**2

    def expand(self, Xu):
        """Applies the expansion transformation to the inducing points

        Args:
            Xu (ndarray): (1, m, 3); Inducing points in the position and orientation space.
                            `m` is the number of inducing points,
                            `3` is the dimension of the space (x, y, angle in radians)
                        
        Returns:
            Xu (ndarray): (mp, 2); Inducing points in input space.
                        `p` is the number of points each inducing point is mapped 
                         to in order to form the FoV.
        """
        x, y, theta = tf.split(Xu, num_or_size_splits=3, axis=2)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        theta = tf.squeeze(theta)

        points = []
        for i in range(-int(np.floor((self.num_side)/2)), int(np.ceil((self.num_side)/2))):
            points.append(tf.linspace([(x + (i * self.length_factor) * tf.cos(theta)) - self.length/2 * tf.cos(theta+np.pi/2), 
                                       (y + (i * self.length_factor) * tf.sin(theta)) - self.length/2 * tf.sin(theta+np.pi/2)], 
                                      [(x + (i * self.length_factor) * tf.cos(theta)) + self.length/2 * tf.cos(theta+np.pi/2), 
                                       (y + (i * self.length_factor) * tf.sin(theta)) + self.length/2 * tf.sin(theta+np.pi/2)], 
                                      self.num_side, axis=1))
        xy = tf.concat(points, axis=1)
        xy = tf.transpose(xy, [2, 1, 0])
        xy = tf.reshape(xy, [-1, 2])
        xy = self._reshape(xy, tf.shape(Xu)[1])
        return xy
    
    def _reshape(self, X, num_inducing):
        """Reorder the inducing points to be in the correct order for aggregation with square FoV.

        Args:
            X (ndarray): (mp, 2); Inducing points in input space. `p` is the number of points each 
                        inducing point is mapped to in order to form the FoV.
                            
        Returns:
            Xu (ndarray): (mp, 2); Reorder inducing points
        """
        X = tf.reshape(X, (num_inducing, -1, self.num_side, self.num_side, 2))
        X = tf.transpose(X, (0, 2, 1, 3, 4))
        X = tf.reshape(X, (-1, 2))
        return X

class IPPTransform(Transform):
    """Transform to model IPP problems

    Usage details: 
        * For point sensing, set `sampling_rate = 2`
        * For continuous sensing, set `sampling_rate > 2` (approx the data collected along the path)
        * For multi-robot case, set `num_robots > 1`
        * For onlineIPP use `update_fixed` to freeze the visited waypoints

    Args:
        sampling_rate (int): Number of points to sample between each pair of inducing points
        distance_budget (float): Distance budget for the path
        num_robots (int): Number of robots
        Xu_fixed (ndarray): (num_robots, num_visited, num_dim); Visited waypoints that don't need to be optimized
        num_dim (int): Dimension of the data collection environment
        sensor_model (Transform): Transform object to expand each inducing point to `p` points 
                                  approximating each sensor's FoV
    """
    def __init__(self, 
                 sampling_rate=2, 
                 distance_budget=None, 
                 num_robots=1,
                 Xu_fixed=None,
                 num_dim=2,
                 sensor_model=None,
                 **kwargs):
        super().__init__(**kwargs)
        if sampling_rate < 2:
            raise ValueError('Sampling rate must be greater than 2.')
        
        self.sampling_rate = sampling_rate
        self.distance_budget = distance_budget
        self.num_robots = num_robots
        self.num_dim = num_dim
        self.sensor_model = sensor_model

        # Disable aggregation if aggregation size was explicitly set to 0
        if self.aggregation_size == 0:
            self.aggregation_size = None
        # Set aggregation size to sampling rate if aggregation size was not set
        # and sampling rate is enabled (greater than 2)
        elif self.aggregation_size is None and sampling_rate > 2:
            self.aggregation_size = sampling_rate
    
        # Initilize variable to store visited waypoints for onlineIPP
        if Xu_fixed is not None:
            self.Xu_fixed = tf.Variable(Xu_fixed, 
                                        shape=tf.TensorShape(None), 
                                        trainable=False)
        else:
            self.Xu_fixed = None

    def update_Xu_fixed(self, Xu_fixed):
        """Function to update the visited waypoints

        Args:
            Xu_fixed (ndarray): numpy array (num_robots, num_visited_waypoints, num_dim)
        """
        self.num_fixed = Xu_fixed.shape[1]
        if self.Xu_fixed is not None:
            self.Xu_fixed.assign(Xu_fixed)
        else:
            self.Xu_fixed = tf.Variable(Xu_fixed, 
                                        shape=tf.TensorShape(None), 
                                        trainable=False)

    def expand(self, Xu, expand_sensor_model=True):
        """Sample points between each pair of inducing points to form the path

        Args:
            Xu (ndarray): (num_robots x num_inducing, num_dim); Inducing points in the num_dim dimensional space
            expand_sensor_model (bool): Only add the fixed inducing points without other sensor/path transforms, 
                                        used for online IPP

        Returns:
            Xu (ndarray): Expansion transformed inducing points
        """
        # If using single-robot offline IPP with point sensing, return inducing points as is.
        if self.sampling_rate == 2 and self.Xu_fixed is None and self.sensor_model is None:
            return Xu
        
        Xu = tf.reshape(Xu, (self.num_robots, -1, self.num_dim))
        
        # If using online IPP, add visited waypoints that won't be optimized anymore
        if self.Xu_fixed is not None:
            Xu = tf.concat([self.Xu_fixed, Xu[:, self.num_fixed:]], axis=1)

        if not expand_sensor_model:
            return tf.reshape(Xu, (-1, self.num_dim))

        # Interpolate additional inducing points between waypoints to approximate 
        # the continuous data sensing model
        if self.sampling_rate > 2:
            Xu = tf.linspace(Xu[:, :-1], Xu[:, 1:], self.sampling_rate)
            Xu = tf.transpose(Xu, perm=[1, 2, 0, 3])
            Xu = tf.reshape(Xu, (self.num_robots, -1, self.num_dim))

        if self.sensor_model is not None:
            Xu = self.sensor_model.expand(Xu)
            return Xu

        Xu = tf.reshape(Xu, (-1, self.num_dim))
        return Xu
    
    def aggregate(self, k):
        """Applies the aggregation transform to kernel matrices. Checks `sensor_model` 
           and uses the appropriate aggregation transform. 

        Args:
            k (tensor): (mp, mp)/(mp, n); Kernel matrix. 
                        `m` is the number of inducing points,
                        `p` is the number of points each inducing point is mapped,
                        `n` is the number of training data points.

        Returns:
            k (tensor): (m, m)/(m, n); Aggregated kernel matrix
        """
        if self.sensor_model is not None:
            return self.sensor_model.aggregate(k)
        else:
            return super().aggregate(k)
        
    def constraints(self, Xu):
        """Computes the distance constraint term that is added to the SGP's optimization function.
        Each robot can be assigned a different distance budget.

        Args:
            Xu (ndarray): Inducing points from which to compute the distance constraints

        Returns:
            loss (float): distance constraint term
        """
        if self.distance_budget is None:
            return 0.
        else:
            Xu = self.expand(Xu, expand_sensor_model=False)
            dist = self.distance(Xu)-self.distance_budget
            dist = tf.reduce_sum(tf.nn.relu(dist))
            loss = -dist*self.constraint_weight
            return loss

    def distance(self, Xu):
        """Computes the distance incured by sequentially visiting the inducing points
        ToDo: Change distance from 2d to nd. Currently limited to 2d 
            to ensure the rotation angle is not included when using
            a square FoV sensor.

        Args:
            Xu (ndarray): Inducing points from which to compute the path lengths

        Returns:
            dist (float): path lengths
        """
        Xu = tf.reshape(Xu, (self.num_robots, -1, self.num_dim))
        dist = tf.norm(Xu[:, 1:, :2] - Xu[:, :-1, :2], axis=-1)
        dist = tf.reduce_sum(dist, axis=1)
        return dist
    

class SquareHeightTransform(Transform):
    """Non-point Transform to model a height-dependent square FoV. Only works for single robot cases. 
    ToDo: Convert from single to multi-robot setup and make it compatible with IPPTransform

    Args:
        num_points (int): Number of points along each side of the FoV
        distance_budget (float): Distance budget for the path
    """
    def __init__(self, num_points, distance_budget=None, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.distance_budget = distance_budget
    
        if self.aggregation_size == 0:
            self.aggregation_size = None
        elif self.aggregation_size is None:
            self.aggregation_size = num_points**2

    def expand(self, Xu):     
        """
        Applies the expansion transform to the inducing points

        Args:
            Xu (ndarray): (m, 3); Inducing points in the 3D position space.
                        `m` is the number of inducing points,
                        `3` is the dimension of the space (x, y, z)
                        
        Returns:
            Xu (ndarray): (mp, 2); Inducing points in input space.
                        `p` is the number of points each inducing point is mapped 
                        to in order to form the FoV.
        """
        x, y, h = tf.split(Xu, num_or_size_splits=3, axis=1)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        h = tf.squeeze(h)

        delta = h / (self.num_points - 1)

        pts = []
        for i in range(self.num_points):
            pts.append(tf.linspace([x - h/2, y - (h/2) + (delta * i)], 
                                   [x + h/2, y - (h/2) + (delta * i)], 
                                   self.num_points, 
                                   axis=1))
        xy = tf.concat(pts, axis=1)
        xy = tf.transpose(xy, [2, 1, 0])
        xy = tf.reshape(xy, [-1, 2])
        xy = self._reshape(xy, tf.shape(Xu)[0])
        return xy

    def _reshape(self, X, num_inducing):
        """Reorder the inducing points to be in the correct order for aggregation with square height FoV

        Args:
            X (ndarray): (mp, 2); Inducing points in input space. `p` is the number of points each 
                        inducing point is mapped to in order to form the FoV.
                            
        Returns:
            Xu (ndarray): (mp, 2); Reorder inducing points
        """
        X = tf.reshape(X, (num_inducing, -1, self.num_points, self.num_points, 2))
        X = tf.transpose(X, (0, 2, 1, 3, 4))
        X = tf.reshape(X, (-1, 2))
        return X
