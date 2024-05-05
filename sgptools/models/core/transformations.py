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

'''
Refer to the following papers for more details:
Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]
Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [Jakkala and Akella, 2024]
'''

try:
    from tensorflow_graphics.math.interpolation import bspline
except:
  pass

import tensorflow as tf
import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


'''
Base class for transformations of the inducing points.
'''
class Transform:
    def __init__(self, 
                 aggregation_size=None, 
                 constraint_weight=1.0,
                 **kwargs):
        self.aggregation_size = aggregation_size
        self.constraint_weight = constraint_weight

    '''
    Expands the inducing points to the desired FoV.
    '''
    def expand(self, Xu):
        return Xu

    '''
    Applies the aggregation transformation to kernel matrices.

    Args:
        k: [ms, ms]/[ms, n] - Kernel matrix. m is the number of inducing points,
                              s is the number of points each inducing point is mapped,
                              n is the number of training data points.

    Returns:
        k: [m, m]/[m, n] - Aggregated kernel matrix.
    '''
    def aggregate(self, k):
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

    '''
    Returns the constraint terms that are added to the elbo.
    '''
    def constraints(self, Xu):
        return 0.

    '''
    Computes the distance incured by sequentially visiting the inducing points.
    '''
    def distance(self, Xu):
        dist = tf.math.reduce_sum(tf.norm(Xu[1:]-Xu[:-1], axis=1))
        return dist
    

'''
Non-point Transform to model a square FoV. Only works for single robot cases. 
ToDo: update expand function to handle multi-robot case
'''
class SquareTransform(Transform):

    '''
    Args:
        length: - length of the square FoV
        num_side: [s] - number of points along each side of the FoV
    '''
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

    '''
    Applies the expansion transformation to the inducing points.

    Args:
        Xu: [1, m, 3] - Inducing points in the position and orientation space.
                        m is the number of inducing points,
                        3 is the dimension of the space (x, y, angle in radians)
                     
    Returns:
        Xu: [mp, 2] - Inducing points in input space.
                      p is the number of points each inducing point is mapped 
                      to in order to form the FoV.
    '''
    def expand(self, Xu):
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
    
    '''
    Reorder the inducing points to be in the correct order for aggregation with square FoV.

    Args:
        X: [mp, 2] - Inducing points in input space. p is the number of points each 
                     inducing point is mapped to in order to form the FoV.
    '''
    def _reshape(self, X, num_inducing):
        X = tf.reshape(X, (num_inducing, -1, self.num_side, self.num_side, 2))
        X = tf.transpose(X, (0, 2, 1, 3, 4))
        X = tf.reshape(X, (-1, 2))
        return X

'''
Transform to model IPP problems. 
-For point sensing, set sampling_rate=2
-For continuous sensing, set sampling_rate>2 (approx the data collected along the path)
-For multi-robot case, set num_robots>1
-For onlineIPP use update_fixed to freeze the visited waypoints
'''
class IPPTransform(Transform):
    '''
    Args:
        sampling_rate: [s] - number of points to sample between each pair of inducing points
        distance_budget: [b] - distance budget for the path
        num_robots: [n] - number of robots
        Xu_fixed: (num_robots, num_visited, num_dim) - visited waypoints that don't need to be optimized
        num_dim: [d] - dimension of the data collection environment
        sensor_model: [Transform] - Transform object to expand each inducing point to p points 
                                      approximating each sensor's FoV.
    '''
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

    '''
    Function to store visited waypoints
    Args:
        Xu_fixed: numpy array (num_robots, num_waypoints, num_dim)
    '''
    def update_Xu_fixed(self, Xu_fixed):
        self.num_fixed = Xu_fixed.shape[1]
        if self.Xu_fixed is not None:
            self.Xu_fixed.assign(Xu_fixed)
        else:
            self.Xu_fixed = tf.Variable(Xu_fixed, 
                                        shape=tf.TensorShape(None), 
                                        trainable=False)

    '''
    Sample points between each pair of inducing points to form the path
    Args:
        Xu: numpy array [num_robots x num_inducing, num_dim] - Inducing points in the num_dim position space.
        expand_sensor_model: Bool - Only add the fixed inducing points without other sensor/path transforms, 
                             used for online IPP. 
    '''
    def expand(self, Xu, expand_sensor_model=True):
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
        if self.sensor_model is not None:
            return self.sensor_model.aggregate(k)
        else:
            return super().aggregate(k)
        
    '''
    Applies a distance constraint to the inducing points.
    '''
    def constraints(self, Xu):
        if self.distance_budget is None:
            return 0.
        else:
            Xu = self.expand(Xu, expand_sensor_model=False)
            dist = self.distance(Xu)-self.distance_budget
            dist = tf.reduce_sum(tf.nn.relu(dist))
            loss = -dist*self.constraint_weight
            return loss

    '''
    Args:
        Xu: [m, d] - Inducing points in the 3D position space.
                     m is the number of inducing points,
                     d is the dimension of the space
    ToDo: Change distance from 2d to nd. Currently limited to 2d 
          to ensure the rotation angle is not included when using
          a square FoV sensor.
    '''
    def distance(self, Xu):
        Xu = tf.reshape(Xu, (self.num_robots, -1, self.num_dim))
        dist = tf.norm(Xu[:, 1:, :2] - Xu[:, :-1, :2], axis=-1)
        dist = tf.reduce_sum(dist, axis=1)
        return dist
    

'''
Applies a mask to the inducing points. 
The mask maps the compact inducing points parametrization to individual points. 
ToDo:
Convert from single to multi-robot setup and make it compatible with IPPTransform
'''
class SquareHeightTransform(Transform):
    
    '''
    Args:
        num_points: [s] - number of points along each side of the FoV
    '''
    def __init__(self, num_points, distance_budget=None, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.distance_budget = distance_budget
    
        if self.aggregation_size == 0:
            self.aggregation_size = None
        elif self.aggregation_size is None:
            self.aggregation_size = num_points**2

    '''
    Applies the expansion transformation to the inducing points.

    Args:
        Xu: [m, 3] - Inducing points in the 3D position space.
                     m is the number of inducing points,
                     3 is the dimension of the space (x, y, z)
                     
    Returns:
        Xu: [ms, 2] - Inducing points in input space.
                      s is the number of points each inducing point is mapped 
                      to in order to form the FoV.
    '''
    def expand(self, Xu):     
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

    '''
    Reorder the inducing points to be in the correct order for aggregation with square FoV.

    Args:
        X: [ms, 2] - Inducing points in input space. s is the number of points each 
                     inducing point is mapped to in order to form the FoV.
    '''
    def _reshape(self, X, num_inducing):
        X = tf.reshape(X, (num_inducing, -1, self.num_points, self.num_points, 2))
        X = tf.transpose(X, (0, 2, 1, 3, 4))
        X = tf.reshape(X, (-1, 2))
        return X

########################################################################################
# Here be dragons
# Transforms for different sensor models (untested code) 
########################################################################################

class SplineIPPTransform(Transform):
    '''
    Args:
        num_points: [s] - number of points along each side of the FoV
    '''
    def __init__(self, num_inducing, sampling_rate, distance_budget=None, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.distance_budget = distance_budget
        self.degree = degree

        max_pos = num_inducing - degree
        self.spline_positions = tf.expand_dims(tf.range(start=0.0, 
                                                        limit=max_pos, 
                                                        delta=max_pos/sampling_rate,
                                                        dtype=tf.float64),
                                               axis=-1)

    def expand(self, Xu):
        Xu = bspline.interpolate(tf.transpose(Xu), self.spline_positions, self.degree, False)
        Xu = tf.squeeze(Xu, axis=1)
        return Xu

    '''
    Args:
        Xu: [m, d] - Inducing points in the 3D position space.
                     m is the number of inducing points,
                     d is the dimension of the space
    '''
    def distance(self, Xu):
        dist = super().distance(self.expand(Xu))
        return dist

    '''
    Applies a distance constraint to the inducing points.
    '''
    def constraints(self, Xu):
        if self.distance_budget is None:
            return 0.
        else:
            dist = self.distance(Xu)-self.distance_budget
            dist = tf.nn.softplus(dist)
            return -dist*100

    def get_path(self, Xu, num_samples):
        max_pos = Xu.shape[0] - self.degree
        spline_positions = tf.expand_dims(tf.range(start=0.0, 
                                                   limit=max_pos, 
                                                   delta=max_pos/num_samples,
                                                   dtype=tf.float64),
                                          axis=-1)
        Xu = bspline.interpolate(tf.transpose(Xu), spline_positions, self.degree, False)
        Xu = tf.squeeze(Xu, axis=1)
        return Xu

# For spatio-temporal models
class MultiRobotIPPTransform(Transform):
    '''
    Args:
        num_points: [s] - number of points along each side of the FoV
    '''
    def __init__(self, num_robots, past_pts=None, distance_budget=None, **kwargs):
        super().__init__(**kwargs)
        self.num_robots = num_robots
        self.distance_budget = distance_budget
        self.past_pts = past_pts

    '''
    Args:
        Xu: [m, d] - Inducing points in the 3D position space.
                     m is the number of inducing points,
                     d is the dimension of the space
    '''
    def distance(self, Xu):
        Xu = self.expand(Xu, add_past_pts=False)
        Xu = tf.reshape(Xu, shape=[-1, self.num_robots, Xu.shape[-1]])
        dist = tf.norm(Xu[1:, :, :2] - Xu[:-1, :, :2], axis=-1)
        dist = tf.reduce_sum(dist, axis=0)
        return dist

    '''
    Applies a distance constraint to the inducing points.
    '''
    def constraints(self, Xu):
        if self.distance_budget is None:
            return 0.
        else:
            dist = self.distance(Xu)-self.distance_budget
            dist = tf.reduce_sum(tf.nn.softplus(dist))
            return -dist*self.constraint_weight

    '''
    Args:
        Xu: [m, d] - Inducing points in the 3D position space.
                     m is the number of inducing points,
                     d is the dimension of the space
    '''
    def expand(self, Xu, add_past_pts=True):
        Xu_s = Xu
        Xu_t = self.inducing_variable_time.Z

        Xu_t = tf.tile(Xu_t, multiples=[1, self.num_robots])
        Xu_t = tf.reshape(Xu_t, shape=[-1, 1])
        Xu = tf.concat([Xu_s, Xu_t], axis=1)

        if self.past_pts is not None and add_past_pts:
            Xu = tf.concat([self.past_pts, Xu], axis=0)

        return Xu

    def inducing2paths(self, Xu):
        Xu = self.expand(Xu, add_past_pts=False).numpy()
        num_time = int(Xu.shape[0]/self.num_robots)

        paths = []
        paths.append(Xu[:self.num_robots, :])
        for i in range(num_time-1):
            dists = pairwise_distances(paths[-1], 
                                       Y=Xu[(i+1)*self.num_robots:(i+2)*self.num_robots, :], 
                                       metric='euclidean')
            _, col_ind = linear_sum_assignment(dists)
            path = Xu[(i+1)*self.num_robots:(i+2)*self.num_robots, :][col_ind].copy()
            paths.append(path)
        return np.array(paths)
    
'''
Applies a mask to the inducing points. 
The mask maps the compact inducing points parametrization to individual points. 
'''
class LineCTScanTransform(Transform):
    '''
    Args:
        num_points: [s] - number of points in each line segment
    '''
    def __init__(self, num_points, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points

        if self.aggregation_size == 0:
            self.aggregation_size = None
        elif self.aggregation_size is None:
            self.aggregation_size = num_points

    '''
    Applies the expansion transformation to the inducing points.

    Args:
        Xu: [m, 1] - Inducing points in the orientation space (angle in radians).
                     m is the number of inducing points.

    Returns:
        Xu: [ms, 2] - Inducing points in input space.
                      s is the number of points each inducing point is mapped 
                      to in order to form each line.
    '''
    def expand(self, Xu):
        # convert angles to lines on the unit circle
        Xu = tf.reshape(Xu, [-1])
        xy = tf.linspace([tf.sin(Xu), tf.cos(Xu)], 
                         [tf.sin(Xu+np.pi), tf.cos(Xu+np.pi)], 
                         self.num_points,
                         axis=1)
        xy = tf.transpose(xy, [2, 1, 0])
        xy = tf.reshape(xy, [-1, 2])
        return xy

'''
Applies a mask to the inducing points. 
The mask maps the compact inducing points parametrization to individual points. 
'''
class FanCTScanTransform(Transform):
    '''
    Args:
        num_points: [s] - number of points in each line segment
    '''
    def __init__(self, num_points, fan_angles, Xu_min=0., Xu_max=np.pi, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.fan_angles = fan_angles
        self.num_fan = fan_angles.shape[0]

        self.Xu_min = Xu_min
        self.Xu_max = Xu_max

        if self.aggregation_size == 0:
            self.aggregation_size = None
        elif self.aggregation_size is None:
            self.aggregation_size = num_points*self.num_fan

    '''
    Applies the expansion transformation to the inducing points.

    Args:
        Xu: [m, 1] - Inducing points in the orientation space (angle in radians).
                     m is the number of inducing points.

    Returns:
        Xu: [ms, 2] - Inducing points in input space.
                      s is the number of points each inducing point is mapped 
                      to in order to form each line.
    '''
    def expand(self, Xu):
        # convert angles to lines on the unit circle
        Xu = tf.reshape(Xu, [-1])
        Xu = self.constrain(Xu)
        
        start = [tf.sin(Xu+np.pi), tf.cos(Xu+np.pi)]
        start = tf.stack(start, axis=1)
        start = tf.tile(start, [1, self.num_fan])
        start = tf.reshape(start, [-1, 2])

        end = Xu[:, None] + self.fan_angles[None, :]
        end = [tf.sin(end), tf.cos(end)]
        end = tf.stack(end, axis=2)
        end = tf.reshape(end, [-1, 2])

        xy = tf.linspace(start, 
                        end, 
                        self.num_points,
                        axis=1)
        xy = tf.reshape(xy, [-1, 2])

        return xy
    
    def constrain(self, Xu):
        '''
        Xu_min = tf.minimum(Xu, self.Xu_min)
        Xu_max = tf.maximum(Xu, self.Xu_max)
        scale = (self.Xu_max - self.Xu_min) / (Xu_max - Xu_min)
        m = self.Xu_min - Xu_min * scale
        Xu = Xu * scale
        Xu = Xu + m
        '''
        return Xu