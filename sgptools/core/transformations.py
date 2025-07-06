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

import tensorflow as tf
from gpflow.config import default_float

float_type = default_float()

import numpy as np
from typing import Optional, Union, Any, Tuple, List  # Import necessary types for type hinting


class Transform:
    """
    Base class for transformations applied to inducing points in sparse Gaussian process models.
    This class defines common interfaces for expanding inducing points (e.g., to model
    sensor fields of view or continuous paths) and aggregating kernel matrices.
    It also provides a base for adding constraint terms to the optimization objective.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]
        - Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [Jakkala and Akella, 2024]
    """

    def __init__(self,
                 aggregation_size: Optional[int] = None,
                 constraint_weight: float = 1.0,
                 **kwargs: Any):
        """
        Initializes the base Transform class.

        Args:
            aggregation_size (Optional[int]): Number of consecutive inducing points to aggregate
                                              when transforming kernel matrices. If None, no aggregation
                                              is performed. Defaults to None.
            constraint_weight (float): A scalar weight that controls the importance of the
                                       constraint terms in the SGP's optimization objective function.
                                       A higher weight means stronger enforcement of constraints.
                                       Defaults to 1.0.
            **kwargs (Any): Additional keyword arguments to be passed to the constructor.
        """
        self.aggregation_size = aggregation_size
        self.constraint_weight = constraint_weight

    def expand(
            self, Xu: Union[np.ndarray,
                            tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
        """
        Applies an expansion transform to the inducing points.
        In this base class, it simply returns the input inducing points unchanged.
        Subclasses should override this method to implement specific expansion logic.

        Args:
            Xu (Union[np.ndarray, tf.Tensor]): The input inducing points.
                                                Shape: (m, d) where `m` is the number of inducing points
                                                and `d` is their dimensionality.

        Returns:
            Union[np.ndarray, tf.Tensor]: The expanded inducing points.
        """
        return Xu

    def aggregate(self, k: tf.Tensor) -> tf.Tensor:
        """
        Applies an aggregation transform to kernel matrices. This is typically used
        to reduce the size of kernel matrices after expansion, by averaging or summing
        over groups of expanded points. This can reduce computational cost for
        matrix inversions (e.g., in `Kuu`).

        Args:
            k (tf.Tensor): The input kernel matrix.
                           Can be (mp, mp) for `Kuu` (square matrix for inducing points
                           against themselves), or (mp, n) for `Kuf` (rectangular matrix
                           for inducing points against training data).
                           `m` is the number of original inducing points.
                           `p` is the number of points each inducing point is mapped to
                           by the expansion transform.
                           `n` is the number of training data points.

        Returns:
            tf.Tensor: The aggregated kernel matrix.
                       Shape: (m, m) if input was (mp, mp), or (m, n) if input was (mp, n).
        """
        if self.aggregation_size is None:
            return k

        # The aggregation logic assumes `k` has leading dimensions that are
        # multiples of `self.aggregation_size`.
        if k.shape[0] == k.shape[
                1]:  # This is K(U, U) or K(U_expanded, U_expanded)
            # Reshape for `tf.nn.avg_pool`: [batch, height, width, channels]
            # Here, we treat the matrix as a 1-channel image.
            k_reshaped = tf.expand_dims(tf.expand_dims(k, axis=0),
                                        axis=-1)  # (1, mp, mp, 1)

            # Apply average pooling. `ksize` and `strides` define the window size
            # and movement for aggregation. This effectively averages blocks.
            k_aggregated = tf.nn.avg_pool(
                k_reshaped,
                ksize=[1, self.aggregation_size, self.aggregation_size, 1],
                strides=[1, self.aggregation_size, self.aggregation_size, 1],
                padding='VALID')
            # Squeeze back to (m, m)
            k = tf.squeeze(k_aggregated, axis=[0, -1])
        else:  # This is K(U, F) or K(U_expanded, F)
            # Reshape for `tf.nn.avg_pool`: (1, mp, n) -> (1, mp, n, 1) if channels are 1
            # Or (1, mp, n) directly for 1D pooling if `n` is treated as a feature dimension.
            # Here, we're pooling along the inducing point dimension.
            k_reshaped = tf.expand_dims(k, axis=0)  # (1, mp, n)
            k_aggregated = tf.nn.avg_pool(
                k_reshaped,
                ksize=[1, self.aggregation_size, 1],  # Pool along height (mp)
                strides=[1, self.aggregation_size, 1],
                padding='VALID')
            # Squeeze back to (m, n)
            k = tf.squeeze(k_aggregated, axis=[0])
        return k

    def constraints(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Computes constraint terms that are added to the SGP's optimization function (ELBO).
        This base implementation returns a zero tensor, implying no constraints by default.
        Subclasses should override this to implement specific constraints (e.g., path length budget).

        Args:
            Xu (tf.Tensor): The inducing points, from which to compute the constraints.
                            Shape: (m, d).

        Returns:
            tf.Tensor: A scalar tensor representing the constraint penalty. Defaults to 0.0.
        """
        return tf.constant(0.0, dtype=float_type)


class IPPTransform(Transform):
    """
    Transform to model Informative Path Planning (IPP) problems for single or multiple robots.
    It handles continuous sensing, non-point fields of view (FoV), and distance constraints.

    * For point sensing (discrete waypoints), set `sampling_rate = 2`.
    * For continuous sensing along paths, set `sampling_rate > 2` to interpolate
        additional points between waypoints for information gathering.
    * For continuous sensing with aggregation for computational efficiency,
        set `sampling_rate > 2` and `aggregate_fov = True`. This averages
        covariances from interpolated points, potentially diminishing solution quality slightly.
    * If using a non-point FoV model (e.g., `SquareTransform`) with continuous sampling,
        only the FoV inducing points are aggregated.
    * For multi-robot scenarios, set `num_robots > 1`.
    * For online IPP where some visited waypoints are fixed, use `update_Xu_fixed`
        to freeze these waypoints from further optimization.
    """

    def __init__(self,
                 sampling_rate: int = 2,
                 distance_budget: Optional[float] = None,
                 num_robots: int = 1,
                 Xu_fixed: Optional[np.ndarray] = None,
                 num_dim: int = 2,
                 sensor_model: Optional[Transform] = None,
                 aggregate_fov: bool = False,
                 **kwargs: Any):
        """
        Initializes the IPPTransform.

        Args:
            sampling_rate (int): Number of points to sample along each segment between two
                                 consecutive inducing points. `sampling_rate=2` implies
                                 only the two endpoints are used (point sensing).
                                 `sampling_rate > 2` implies continuous sensing via interpolation.
                                 Must be $\ge 2$. Defaults to 2.
            distance_budget (Optional[float]): The maximum allowable total path length for each robot.
                                               If None, no distance constraint is applied. Defaults to None.
            num_robots (int): The number of robots or agents involved in the IPP problem. Defaults to 1.
            Xu_fixed (Optional[np.ndarray]): (num_robots, num_visited, num_dim);
                                            An array of pre-defined, fixed waypoints that should
                                            not be optimized (e.g., already visited locations in online IPP).
                                            If None, all waypoints are optimizable. Defaults to None.
            num_dim (int): The dimensionality of the inducing points (e.g., 2 for (x,y), 3 for (x,y,angle)).
                           Defaults to 2.
            sensor_model (Optional[Transform]): A `Transform` object that defines a non-point
                                                Field of View (FoV) for the sensor (e.g., `SquareTransform`).
                                                If None, a point sensor model is assumed. Defaults to None.
            aggregate_fov (bool): If True, and `sampling_rate > 2` (continuous sensing is enabled),
                                  or if `sensor_model` is provided, aggregation will be enabled.
                                  This reduces computation by averaging expanded points' covariances.
                                  Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to the base `Transform` constructor.

        Raises:
            ValueError: If `sampling_rate` is less than 2.

        Usage:
            ```python
            # Single robot, point sensing
            transform_point = IPPTransform(num_robots=1, num_dim=2, sampling_rate=2)

            # Single robot, continuous sensing
            transform_continuous = IPPTransform(num_robots=1, num_dim=2, sampling_rate=10)

            # Multi-robot, continuous sensing with distance budget
            transform_multi_budget = IPPTransform(num_robots=2, num_dim=2, sampling_rate=5, distance_budget=50.0, constraint_weight=100.0)
            ```
        """
        super().__init__(**kwargs)
        if sampling_rate < 2:
            raise ValueError(
                'Sampling rate must be greater than or equal to 2.')

        self.sampling_rate = sampling_rate
        self.distance_budget = distance_budget
        self.num_robots = num_robots
        self.num_dim = num_dim
        self.sensor_model = sensor_model

        # Enable aggregation if `aggregate_fov` is True, potentially leveraging the sensor_model's aggregation
        if aggregate_fov:
            if self.sensor_model is not None:
                # If a sensor model exists, let it handle its own aggregation settings.
                # This assumes `sensor_model` has an `enable_aggregation` method.
                if hasattr(self.sensor_model, 'enable_aggregation'):
                    self.sensor_model.enable_aggregation()
            elif self.sampling_rate > 2:
                # If no specific sensor model but continuous sensing, aggregate based on sampling rate.
                self.aggregation_size = self.sampling_rate

        # Initialize TensorFlow Variable for fixed waypoints if provided, for online IPP.
        if Xu_fixed is not None:
            # Store number of fixed waypoints per robot
            self.num_fixed = Xu_fixed.shape[1]  
            self.Xu_fixed = tf.Variable(
                Xu_fixed,
                shape=tf.TensorShape(None),
                trainable=False,  # Fixed points are not optimized
                dtype=float_type)
        else:
            self.Xu_fixed = None

    def update_Xu_fixed(self, Xu_fixed: np.ndarray) -> None:
        """
        Updates the set of visited (fixed) waypoints for online IPP scenarios.
        These waypoints will not be optimized in subsequent steps.

        Args:
            Xu_fixed (np.ndarray): A NumPy array of shape (num_robots, num_visited_waypoints, num_dim)
                                   representing the new set of fixed waypoints.
        """
        # Store number of fixed waypoints per robot
        self.num_fixed = Xu_fixed.shape[1]  
        if self.Xu_fixed is not None:
            self.Xu_fixed.assign(tf.constant(Xu_fixed, dtype=float_type))
        else:
            self.Xu_fixed = tf.Variable(Xu_fixed,
                                        shape=tf.TensorShape(None),
                                        trainable=False,
                                        dtype=float_type)

    def expand(self,
               Xu: tf.Tensor,
               expand_sensor_model: bool = True) -> tf.Tensor:
        """
        Applies the expansion transform to the inducing points based on the IPP settings.
        This can involve:
        1. Adding fixed (already visited) waypoints.
        2. Interpolating points between waypoints for continuous sensing.
        3. Expanding each point into a sensor's Field of View (FoV) if a `sensor_model` is present.

        Args:
            Xu (tf.Tensor): The current set of optimizable inducing points.
                            Shape: (num_robots * num_optimizable_waypoints, num_dim).
                            Note: `num_dim` might include angle if `sensor_model` requires it.
            expand_sensor_model (bool): If True, applies the `sensor_model`'s expansion
                                        (if a `sensor_model` is configured). If False,
                                        only the path interpolation and fixed point handling
                                        are performed, useful for internal calculations like distance.
                                        Defaults to True.

        Returns:
            tf.Tensor: The expanded inducing points, ready for kernel computations.
                       Shape: (total_expanded_points, d_output), where `d_output`
                       is typically 2 for (x,y) coordinates used in the kernel.
        """
        # If using single-robot offline IPP with point sensing, return inducing points as is.
        if self.sampling_rate == 2 and self.Xu_fixed is None and self.sensor_model is None:
            return Xu

        # Reshape Xu to (num_robots, num_waypoints_per_robot, num_dim)
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

        # If a sensor model is defined and `expand_sensor_model` is True,
        # apply the sensor model's expansion for each robot.
        if self.sensor_model is not None:
            Xu_ = []
            for i in range(self.num_robots):
                Xu_.append(self.sensor_model.expand(Xu[i]))
            Xu = tf.concat(Xu_, axis=0)
            return Xu

        Xu = tf.reshape(Xu, (-1, self.num_dim))
        return Xu

    def aggregate(self, k: tf.Tensor) -> tf.Tensor:
        """
        Applies the aggregation transform to kernel matrices.
        If a `sensor_model` is defined, it delegates aggregation to the sensor model.
        Otherwise, it uses the base class's aggregation logic (which depends on `self.aggregation_size`).

        Args:
            k (tf.Tensor): The input kernel matrix (e.g., K_expanded_expanded, K_expanded_training).

        Returns:
            tf.Tensor: The aggregated kernel matrix.
        """
        if self.sensor_model is not None:
            return self.sensor_model.aggregate(k)
        else:
            return super().aggregate(k)

    def constraints(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Computes the distance constraint term that is added to the SGP's optimization function.
        Each robot's path length is constrained by `distance_budget`. A penalty is applied
        if the path length exceeds the budget.

        Args:
            Xu (tf.Tensor): The inducing points (waypoints) from which to compute path lengths.
                            Shape: (num_robots * num_waypoints, num_dim).

        Returns:
            tf.Tensor: A scalar tensor representing the total distance constraint penalty.
                       This value is negative, and its magnitude increases with constraint violation.
        """
        if self.distance_budget is None:
            return tf.constant(0.0, dtype=float_type)  # No distance constraint
        else:
            # Expand Xu without sensor model to get the true path points for distance calculation.
            # Xu is the optimizable part; self.expand will add fixed points if any.
            Xu_for_distance = self.expand(Xu, expand_sensor_model=False)

            # Calculate distances for each robot's path
            individual_robot_distances = self.distance(Xu_for_distance)

            # Compute the positive violation for each robot's path
            violations = individual_robot_distances - self.distance_budget

            # Apply ReLU to ensure only positive violations contribute to the penalty
            # Sum all violations and apply the constraint weight
            penalty = -tf.reduce_sum(
                tf.nn.relu(violations)) * self.constraint_weight
            return penalty

    def distance(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Computes the total path length(s) incurred by sequentially visiting the inducing points.
        For multiple robots, returns a tensor of individual path lengths.

        Args:
            Xu (tf.Tensor): The inducing points (waypoints) from which to compute the path lengths.
                            Shape: (total_waypoints, num_dim). This input is typically already
                            expanded to include fixed points if `Xu_fixed` is used.

        Returns:
            tf.Tensor: A scalar tensor if `num_robots=1`, or a 1D tensor of floats
                       (shape: `(num_robots,)`) representing individual path lengths.
        """
        # Reshape to (num_robots, num_waypoints_per_robot, num_dim)
        Xu_reshaped = tf.reshape(Xu, (self.num_robots, -1, self.num_dim))

        if self.sensor_model is not None:
            # If a sensor model is present, delegate distance calculation to it,
            # as it might have specific logic for its FoV's contribution to distance.
            dists: List[tf.Tensor] = []
            for i in range(self.num_robots):
                # Pass each robot's path (which includes position and potentially angle)
                dists.append(self.sensor_model.distance(Xu_reshaped[i]))
            return tf.concat(
                dists, axis=0)  # Concatenate distances if multiple robots
        else:
            # For point/continuous sensing without a special FoV model:
            # Calculate Euclidean distance between consecutive waypoints.
            # Assuming first two dimensions are (x,y) for distance calculation.
            # `Xu_reshaped[:, 1:, :2]` are points from the second to last.
            # `Xu_reshaped[:, :-1, :2]` are points from the first to second to last.
            segment_distances = tf.norm(Xu_reshaped[:, 1:, :2] -
                                        Xu_reshaped[:, :-1, :2],
                                        axis=-1)
            total_distances = tf.reduce_sum(
                segment_distances, axis=1)  # Sum distances for each robot
            return total_distances


class SquareTransform(Transform):
    """
    Non-point Transform to model a square Field of View (FoV) for a sensor.
    This transform expands each inducing point (waypoint with position and orientation)
    into a grid of points approximating a square area, which is then used in kernel computations.
    This typically applies to single-robot cases as part of an `IPPTransform`.
    """

    def __init__(self,
                 side_length: float,
                 pts_per_side: int,
                 aggregate_fov: bool = False,
                 **kwargs: Any):
        """
        Initializes the SquareTransform for a square FoV.

        Args:
            side_length (float): The side length of the square FoV.
            pts_per_side (int): The number of points to sample along each side of the square.
                                A `pts_per_side` of 3 will create a 3x3 grid of 9 points to approximate the FoV.
            aggregate_fov (bool): If True, aggregation will be enabled for the expanded FoV points.
                                  This averages covariances from the FoV points to reduce computational cost.
                                  Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to the base `Transform` constructor.

        Usage:
            ```python
            # Create a square FoV of side length 10.0, approximated by a 5x5 grid of points
            square_fov_transform = SquareTransform(length=10.0, pts_per_side=5, aggregate_fov=True)
            ```
        """
        super().__init__(**kwargs)
        self.side_length = side_length
        self.pts_per_side = pts_per_side
        # Calculate the spacing between points along each side
        self.side_length_factor = side_length / (self.pts_per_side)

        if aggregate_fov:
            self.enable_aggregation()

    def enable_aggregation(self, size: Optional[int] = None) -> None:
        """
        Enables FoV covariance aggregation. This reduces the covariance matrix inversion
        cost by effectively reducing the covariance matrix size.

        Args:
            size (Optional[int]): If None, all the interpolated inducing points within the FoV
                                  (i.e., `pts_per_side^2` points) are aggregated into a single aggregated point.
                                  Alternatively, the number of inducing points to aggregate can be
                                  explicitly defined using this variable (e.g., if a custom
                                  aggregation strategy is desired that groups `size` points).
                                  Defaults to None.
        """
        if size is None:
            self.aggregation_size = self.pts_per_side**2  # Aggregate all points within a FoV
        else:
            self.aggregation_size = size

    def expand(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Applies the expansion transformation to the inducing points, modeling a square FoV.
        Each input inducing point, which includes position (x, y) and orientation (theta),
        is expanded into a grid of `pts_per_side` x `pts_per_side` points representing the FoV.

        Args:
            Xu (tf.Tensor): Inducing points in the position and orientation space.
                            Shape: (m, 3) where `m` is the number of inducing points,
                            and `3` corresponds to (x, y, angle in radians).
                        
        Returns:
            tf.Tensor: The expanded inducing points in 2D input space (x,y).
                       Shape: (m * pts_per_side * pts_per_side, 2).
                       `m` is the number of original inducing points.
                       `pts_per_side * pts_per_side` is the number of points each inducing
                       point is mapped to in order to form the FoV.
        """
        # Split Xu into x, y coordinates and orientation (theta)
        x_coords, y_coords, angles = tf.split(Xu, num_or_size_splits=3, axis=1)
        x = tf.reshape(x_coords, [
            -1,
        ])  # Flatten to (m,)
        y = tf.reshape(y_coords, [
            -1,
        ])
        theta = tf.reshape(angles, [
            -1,
        ])

        points: List[tf.Tensor] = []
        # Iterate to create `pts_per_side` lines forming the square grid.
        # The loop runs from -floor(pts_per_side/2) to ceil(pts_per_side/2) to center the grid.
        for i in range(-int(np.floor((self.pts_per_side - 1) / 2)),
                       int(np.ceil((self.pts_per_side - 1) / 2)) + 1):
            # Calculate start and end points for each line segment of the square grid.
            # `(i * self.side_length_factor)` shifts the line perpendicular to the orientation `theta`.
            # `self.side_length/2` extends the line segment along the orientation `theta`.

            # Start point (x,y) for the current line
            start_x = x + (i * self.side_length_factor) * tf.cos(
                theta + np.pi / 2) - self.side_length / 2 * tf.cos(theta)
            start_y = y + (i * self.side_length_factor) * tf.sin(
                theta + np.pi / 2) - self.side_length / 2 * tf.sin(theta)

            # End point (x,y) for the current line
            end_x = x + (i * self.side_length_factor) * tf.cos(
                theta + np.pi / 2) + self.side_length / 2 * tf.cos(theta)
            end_y = y + (i * self.side_length_factor) * tf.sin(
                theta + np.pi / 2) + self.side_length / 2 * tf.sin(theta)

            # Stack start and end points for linspace
            line_starts = tf.stack([start_x, start_y], axis=-1)  # (m, 2)
            line_ends = tf.stack([end_x, end_y], axis=-1)  # (m, 2)

            # Generate `self.pts_per_side` points along each line segment.
            # `axis=1` ensures interpolation is done column-wise for each (start, end) pair.
            # The result is (m, pts_per_side, 2) for each `i`.
            points.append(
                tf.linspace(line_starts, line_ends, self.pts_per_side, axis=1))

        # Concatenate all generated line segments.
        # `tf.concat` will stack them along a new axis, forming (num_lines, m, pts_per_side, 2)
        xy = tf.concat(
            points, axis=1
        )  # (m, pts_per_side * pts_per_side, 2) after the transpose in the original code.

        xy = tf.reshape(xy, (-1, 2))
        return xy

    def distance(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Computes the Euclidean distance incurred by sequentially visiting the inducing points.
        For a Square FoV, the distance is typically only based on the (x,y) movement,
        ignoring the angle.

        Args:
            Xu (tf.Tensor): Inducing points.
                            Shape: (m, 3) where `m` is the number of inducing points,
                            and `3` corresponds to (x, y, angle).

        Returns:
            tf.Tensor: A scalar tensor representing the total path length.
        """
        # Reshape to (number_of_points, 3) and take only the (x,y) coordinates
        Xu_xy = tf.reshape(
            Xu, (-1, self.num_dim))[:, :2]  # Assuming num_dim is 3 (x,y,angle)

        if Xu_xy.shape[0] < 2:
            return tf.constant(0.0, dtype=float_type)

        # Calculate Euclidean distance between consecutive (x,y) points
        segment_distances = tf.norm(Xu_xy[1:] - Xu_xy[:-1], axis=-1)
        total_distance = tf.reduce_sum(segment_distances, axis=0)
        return total_distance


class SquareHeightTransform(Transform):
    """
    Non-point Transform to model a height-dependent square Field of View (FoV).
    The size of the square FoV changes with the 'height' (z-dimension) of the sensor.
    This transform expands each inducing point (waypoint with x, y, z coordinates)
    into a grid of points approximating a square area whose size depends on 'z'.
    """

    def __init__(self,
                 pts_per_side: int,
                 aggregate_fov: bool = False,
                 **kwargs: Any):
        """
        Initializes the SquareHeightTransform for a height-dependent square FoV.

        Args:
            pts_per_side (int): The number of points to sample along each side of the square FoV.
                            A `pts_per_side` of 3 will create a 3x3 grid of 9 points to approximate the FoV.
            aggregate_fov (bool): If True, aggregation will be enabled for the expanded FoV points.
                                  This averages covariances from the FoV points to reduce computational cost.
                                  Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to the base `Transform` constructor.

        Usage:
            ```python
            # Create a height-dependent square FoV approximated by a 7x7 grid
            square_height_fov_transform = SquareHeightTransform(pts_per_side=7, aggregate_fov=True)
            ```
        """
        super().__init__(**kwargs)
        self.pts_per_side = pts_per_side

        if aggregate_fov:
            self.enable_aggregation()

    def enable_aggregation(self, size: Optional[int] = None) -> None:
        """
        Enables FoV covariance aggregation, which reduces the covariance matrix inversion
        cost by effectively reducing the covariance matrix size.

        Args:
            size (Optional[int]): If None, all the interpolated inducing points within the FoV
                                  (i.e., `pts_per_side^2` points) are aggregated into a single aggregated point.
                                  Alternatively, the number of inducing points to aggregate can be
                                  explicitly defined using this variable. Defaults to None.
        """
        if size is None:
            self.aggregation_size = self.pts_per_side**2
        else:
            self.aggregation_size = size

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
        x = tf.reshape(x, [
            -1,
        ])
        y = tf.reshape(y, [
            -1,
        ])
        h = tf.reshape(h, [
            -1,
        ])

        delta = h / (self.pts_per_side - 1)

        pts = []
        for i in range(self.pts_per_side):
            line_starts = [x - h / 2, y - (h / 2) + (delta * i)]
            line_ends = [x + h / 2, y - (h / 2) + (delta * i)]
            pts.append(
                tf.linspace(line_starts, line_ends, self.pts_per_side, axis=1))
        xy = tf.concat(pts, axis=1)
        xy = tf.transpose(xy, [2, 1, 0])
        xy = tf.reshape(xy, [-1, 2])
        xy = self._reshape(xy, tf.shape(Xu)[0])
        return xy

    def _reshape(self, X: tf.Tensor, num_inducing: tf.Tensor) -> tf.Tensor:
        """
        Reorders the expanded inducing points.

        Args:
            X (tf.Tensor): Expanded inducing points.
            num_inducing (tf.Tensor): Original number of inducing points.

        Returns:
            tf.Tensor: Reordered expanded inducing points.
        """
        X = tf.reshape(
            X, (num_inducing, -1, self.pts_per_side, self.pts_per_side, 2))
        X = tf.transpose(X, (0, 2, 1, 3, 4))
        X = tf.reshape(X, (-1, 2))
        return X

    def distance(self, Xu: tf.Tensor) -> tf.Tensor:
        """
        Computes the Euclidean distance incurred by sequentially visiting the inducing points.
        For a height-dependent Square FoV, the distance is typically only based on the
        (x,y,z) movement.

        Args:
            Xu (tf.Tensor): Inducing points.
                            Shape: (m, 3) where `m` is the number of inducing points,
                            and `3` corresponds to (x, y, z).

        Returns:
            tf.Tensor: A scalar tensor representing the total path length.
        """
        # Reshape to (number_of_points, 3)
        Xu_xyz = tf.reshape(Xu, (-1, 3))

        # Calculate Euclidean distance between consecutive (x,y,z) points
        segment_distances = tf.norm(Xu_xyz[1:] - Xu_xyz[:-1], axis=-1)
        total_distance = tf.reduce_sum(segment_distances, axis=0)
        return total_distance
