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

import numpy as np
from matplotlib import path
from skimage.draw import line
from .misc import get_inducing_pts
from sklearn.preprocessing import StandardScaler
from hkb_diamondsquare.DiamondSquare import diamond_square

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 900000000

from typing import List, Tuple, Optional, Any


def remove_polygons(
        X: np.ndarray, Y: np.ndarray,
        polygons: List[path.Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes points that fall inside a list of matplotlib Path polygons.

    Args:
        X (np.ndarray): (N,); Array of x-coordinates.
        Y (np.ndarray): (N,); Array of y-coordinates.
        polygons (List[path.Path]): A list of `matplotlib.path.Path` objects.
                                     Points within these polygons will be removed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D NumPy arrays:
                                       (filtered_X_coordinates, filtered_Y_coordinates).

    Usage:
        ```python
        import matplotlib.path as mpath
        import numpy as np

        # Example points
        X_coords = np.array([0, 1, 2, 3, 4, 5])
        Y_coords = np.array([0, 1, 2, 3, 4, 5])

        # Define a square polygon (points inside will be removed)
        polygon_vertices = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])
        square_polygon = mpath.Path(polygon_vertices)

        filtered_X, filtered_Y = remove_polygons(X_coords, Y_coords, [square_polygon])
        ```
    """
    points = np.array([X.flatten(), Y.flatten()]).T
    for polygon in polygons:
        p = path.Path(polygon)
        points = points[~p.contains_points(points)]
    return points[:, 0], points[:, 1]


def remove_circle_patches(
        X: np.ndarray, Y: np.ndarray,
        circle_patches: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes points that fall inside a list of matplotlib Circle patches.

    Note: This function assumes that the `circle_patch` objects have a `contains_points` method,
    similar to `matplotlib.patches.Circle` or `matplotlib.path.Path`.

    Args:
        X (np.ndarray): (N,); Array of x-coordinates.
        Y (np.ndarray): (N,); Array of y-coordinates.
        circle_patches (List[Any]): A list of objects representing circle patches.
                                     Each object must have a `contains_points(points)` method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D NumPy arrays:
                                       (filtered_X_coordinates, filtered_Y_coordinates).

    Usage:
        ```python
        import numpy as np
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        # Example points
        X_coords = np.array([0, 1, 2, 3, 4, 5])
        Y_coords = np.array([0, 1, 2, 3, 4, 5])

        # Define a circle patch centered at (2,2) with radius 1.5
        circle = Circle((2, 2), 1.5)

        filtered_X, filtered_Y = remove_circle_patches(X_coords, Y_coords, [circle])
        ```
    """
    points = np.array([X.flatten(), Y.flatten()]).T
    for circle_patch in circle_patches:
        points = points[~circle_patch.contains_points(points)]
    return points[:, 0], points[:, 1]


def point_pos(point: np.ndarray, d: float, theta: float) -> np.ndarray:
    """
    Generates a new point at a specified distance `d` and angle `theta`
    (in radians) from an existing point. This function applies the
    transformation to multiple points simultaneously.

    Args:
        point (np.ndarray): (N, 2); Array of original 2D points (x, y).
        d (float): The distance from the original point to the new point.
        theta (float): The angle in radians for the direction of displacement.

    Returns:
        np.ndarray: (N, 2); An array of new points after displacement.

    Usage:
        ```python
        import numpy as np

        # Example points (N=2)
        initial_points = np.array([[0.0, 0.0], [1.0, 1.0]])
        # Displace by distance 5.0 at angle pi/4 (45 degrees)
        new_points = point_pos(initial_points, 5.0, np.pi/4)
        # Expected:
        # New points:
        # [[3.53553391 3.53553391]
        #  [4.53553391 4.53553391]]
        ```
    """
    return np.c_[point[:, 0] + d * np.cos(theta),
                 point[:, 1] + d * np.sin(theta)]


def prep_tif_dataset(dataset_path: str,
                     dim_max: int = 2500,
                     verbose: bool = True) -> np.ndarray:
    """
    Loads and preprocesses a dataset from a GeoTIFF (.tif) file.
    The function handles downsampling for large files and replaces NoData values (-999999.0) with NaN.

    For very large .tif files, it's recommended to downsample them externally using GDAL:
    `gdalwarp -tr 50 50 <input>.tif <output>.tif`

    Args:
        dataset_path (str): Path to the GeoTIFF dataset file.
        dim_max (int): Maximum allowed dimension (width or height) for the loaded dataset.
                       If either dimension exceeds `dim_max`, the image will be downsampled
                       to fit, maintaining aspect ratio. Defaults to 2500.
        verbose (bool): If `True`, print details about loading and downsampling. Defaults to True.

    Returns:
       np.ndarray: (H, W); The preprocessed 2D NumPy array representing the dataset,
                   with NoData values converted to NaN.

    Usage:
        ```python
        # Assuming 'path/to/your/dataset.tif' exists
        # from sgptools.utils.data import prep_tif_dataset
        # dataset_array = prep_tif_dataset('path/to/your/dataset.tif', dim_max=1000)
        ```
    """
    data = PIL.Image.open(dataset_path)
    data_array = np.array(data)
    if verbose:
        print(
            f"Loaded dataset from {dataset_path} with shape {data_array.shape}"
        )

    downsample_factor = np.ceil(np.max(data_array.shape) / dim_max).astype(int)
    if downsample_factor <= 1:
        downsample_factor = 1
    elif verbose:
        print(
            f'Downsampling by a factor of {downsample_factor} to fit the maximum dimension of {dim_max}'
        )

    # Downsample and convert to float, replace specific NoData value with NaN
    data_array = data_array[::downsample_factor, ::downsample_factor].astype(
        float)
    data_array[np.where(data_array == -999999.0)] = np.nan
    return data_array


def prep_synthetic_dataset(shape: Tuple[int, int] = (1000, 1000),
                           min_height: float = 0.0,
                           max_height: float = 30.0,
                           roughness: float = 0.5,
                           random_seed: Optional[int] = None,
                           **kwargs: Any) -> np.ndarray:
    """
    Generates a 2D synthetic elevation (or similar) dataset using the diamond-square algorithm.

    Reference: [https://github.com/buckinha/DiamondSquare](https://github.com/buckinha/DiamondSquare)

    Args:
        shape (Tuple[int, int]): (width, height); The dimensions of the generated grid. Defaults to (1000, 1000).
        min_height (float): Minimum allowed value in the generated data. Defaults to 0.0.
        max_height (float): Maximum allowed value in the generated data. Defaults to 30.0.
        roughness (float): Controls the fractal dimension of the generated terrain. Higher
                           values produce rougher terrain. Defaults to 0.5.
        random_seed (Optional[int]): Seed for reproducibility of the generated data. Defaults to None.
        **kwargs: Additional keyword arguments passed directly to the `diamond_square` function.

    Returns:
       np.ndarray: (height, width); The generated 2D synthetic dataset.

    Usage:
        ```python
        # from sgptools.utils.data import prep_synthetic_dataset
        # synthetic_data = prep_synthetic_dataset(shape=(256, 256), roughness=0.7, random_seed=42)
        ```
    """
    data = diamond_square(shape=shape,
                          min_height=min_height,
                          max_height=max_height,
                          roughness=roughness,
                          random_seed=random_seed,
                          **kwargs)
    return data.astype(float)


class Dataset:
    """
    A class to load, preprocess, and manage access to a dataset for sensor placement
    and informative path planning tasks.

    It handles the following operations:
    
    * Loading from a GeoTIFF file, loading from a numpy array, and generating a synthetic dataset.
    * Sampling training, testing, and candidate points from valid (non-NaN) locations.
    * Standardizing both the input coordinates (X) and the labels (y) using `StandardScaler`.
    * Providing methods to retrieve different subsets of the data (train, test, candidates)
    and to sample sensor data at specified locations or along a path.

    The dataset is expected to be a 2D array where each element represents a label
    (e.g., elevation, temperature, environmental reading).
    """

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 num_train: int = 1000,
                 num_test: int = 2500,
                 num_candidates: int = 150,
                 verbose: bool = True,
                 data=None,
                 dtype=np.float64,
                 **kwargs: Any):
        """
        Initializes the Dataset class.

        Args:
            dataset_path (Optional[str]): Path to the dataset file (e.g., '.tif'). If None,
                                          a synthetic dataset will be generated. Defaults to None.
                                          Alternatively, pass an array of data to the constructor
                                          with the `data` argument to use a custom dataset.
            num_train (int): Number of training points to sample from the dataset. Defaults to 1000.
            num_test (int): Number of testing points to sample from the dataset. Defaults to 2500.
            num_candidates (int): Number of candidate points for potential sensor placements
                                  to sample from the dataset. Defaults to 150.
            verbose (bool): If `True`, print details about dataset loading, sampling, and preprocessing.
                            Defaults to True.
            data (Optional[np.ndarray]): (height, width, d); 2D n-dimensional array of data.
            dtype (Optional[np.dtype]): The type of the output arrays. If dtype is not given, 
                                        it will be set to np.float64.
            **kwargs: Additional keyword arguments passed to `prep_tif_dataset` or `prep_synthetic_dataset`.
        """
        self.verbose = verbose
        self.dtype = dtype

        # Load/Create the data
        if data is not None:
            self.y = data
        elif dataset_path is not None:
            self.y = prep_tif_dataset(dataset_path=dataset_path,
                                      verbose=verbose,
                                      **kwargs)
        else:
            self.y = prep_synthetic_dataset(**kwargs)

        # Store original dimensions for reshaping
        w, h = self.y.shape[0], self.y.shape[1]
        if self.verbose:
            print(f"Original dataset shape: {self.y.shape}")

        # Get valid points (non-NaN labels)
        mask = np.where(np.isfinite(self.y))
        X_valid_pixel_coords = np.column_stack((mask[0], mask[1]))

        # Sample training, testing, and candidate points from valid pixel coordinates
        # `get_inducing_pts` with random=True is used for random sampling
        X_train_pixel_coords = get_inducing_pts(X_valid_pixel_coords,
                                                num_train,
                                                random=True)
        y_train_raw = self.y[X_train_pixel_coords[:, 0],
                             X_train_pixel_coords[:, 1]].reshape(-1, 1)

        # If num_test is equal to dataset size, return test data in original order, enables plotting with imshow
        if self.y.shape[0] * self.y.shape[1] == num_test:
            X_test_pixel_coords = X_valid_pixel_coords
            y_test_raw = self.y.reshape(-1, 1)
        else:
            X_test_pixel_coords = get_inducing_pts(X_valid_pixel_coords,
                                                   num_test,
                                                   random=True)
            y_test_raw = self.y[X_test_pixel_coords[:, 0],
                                X_test_pixel_coords[:, 1]].reshape(-1, 1)

        X_candidates_pixel_coords = get_inducing_pts(X_valid_pixel_coords,
                                                     num_candidates,
                                                     random=True)

        # Standardize dataset X coordinates (pixel coords to normalized space)
        self.X_scaler = StandardScaler()
        self.X_scaler.fit(X_train_pixel_coords)

        # Adjust X_scaler's variance/scale to ensure uniform scaling across dimensions
        # and to scale the data to have an extent of at least 10.0 in each dimension.
        # This ensures consistency and prevents issues with very small scales.
        ind = np.argmax(self.X_scaler.var_)
        self.X_scaler.var_ = np.ones_like(
            self.X_scaler.var_) * self.X_scaler.var_[ind]
        self.X_scaler.scale_ = np.ones_like(
            self.X_scaler.scale_) * self.X_scaler.scale_[ind]
        self.X_scaler.scale_ /= 10.0  # Scale to ensure an extent of ~10 units

        self.X_train = self.X_scaler.transform(X_train_pixel_coords)
        self.X_train = self.X_train.astype(self.dtype)
        self.X_test = self.X_scaler.transform(X_test_pixel_coords)
        self.X_test = self.X_test.astype(self.dtype)
        self.candidates = self.X_scaler.transform(X_candidates_pixel_coords)
        self.candidates = self.candidates.astype(self.dtype)

        # Standardize dataset labels (y values)
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(y_train_raw)

        self.y_train = self.y_scaler.transform(y_train_raw)
        self.y_train = self.y_train.astype(self.dtype)
        self.y_test = self.y_scaler.transform(y_test_raw)
        self.y_test = self.y_test.astype(self.dtype)

        # Transform the entire dataset's labels for consistency
        self.y = self.y_scaler.transform(self.y.reshape(-1, 1)).reshape(w, h)
        self.y = self.y.astype(self.dtype)

        if self.verbose:
            print(
                f"Training data shapes (X, y): {self.X_train.shape}, {self.y_train.shape}"
            )
            print(
                f"Testing data shapes (X, y): {self.X_test.shape}, {self.y_test.shape}"
            )
            print(f"Candidate data shape (X): {self.candidates.shape}")
            print("Dataset loaded and preprocessed successfully.")

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the preprocessed training data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                           - X_train (np.ndarray): (num_train, 2); Normalized training input features.
                                           - y_train (np.ndarray): (num_train, 1); Standardized training labels.

        Usage:
            ```python
            # dataset_obj = Dataset(...)
            # X_train, y_train = dataset_obj.get_train()
            ```
        """
        return self.X_train, self.y_train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the preprocessed testing data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                           - X_test (np.ndarray): (num_test, 2); Normalized testing input features.
                                           - y_test (np.ndarray): (num_test, 1); Standardized testing labels.

        Usage:
            ```python
            # dataset_obj = Dataset(...)
            # X_test, y_test = dataset_obj.get_test()
            ```
        """
        return self.X_test, self.y_test

    def get_candidates(self) -> np.ndarray:
        """
        Retrieves the preprocessed candidate locations for sensor placement.

        Returns:
            np.ndarray: (num_candidates, 2); Normalized candidate locations.

        Usage:
            ```python
            # dataset_obj = Dataset(...)
            # candidates = dataset_obj.get_candidates()
            ```
        """
        return self.candidates

    def get_sensor_data(
            self,
            locations: np.ndarray,
            continuous_sening: bool = False,
            max_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples sensor data (labels) at specified normalized locations.
        Can simulate discrete point sensing or continuous path sensing by interpolation.

        Args:
            locations (np.ndarray): (N, 2); Array of locations (normalized x, y coordinates)
                                    where sensor data is to be sampled.
            continuous_sening (bool): If `True`, interpolates additional points between
                                      the given `locations` to simulate sensing along a path.
                                      Defaults to `False`.
            max_samples (int): Maximum number of samples to return if `continuous_sening`
                               results in too many points. If the number of interpolated
                               points exceeds `max_samples`, a random subset will be returned.
                               Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                           - sampled_locations (np.ndarray): (M, 2); Normalized locations
                                                                             where sensor data was effectively sampled.
                                           - sampled_data (np.ndarray): (M, 1); Standardized sensor data
                                                                        sampled at these locations.
                                           Returns empty arrays if no valid data points are found.

        Usage:
            ```python
            # dataset_obj = Dataset(...)
            # X_path_normalized = np.array([[0.1, 0.2], [0.5, 0.7], [0.9, 0.8]])
            # # Discrete sensing
            # sensed_X_discrete, sensed_y_discrete = dataset_obj.get_sensor_data(X_path_normalized)
            # # Continuous sensing with interpolation
            # sensed_X_continuous, sensed_y_continuous = dataset_obj.get_sensor_data(X_path_normalized, continuous_sening=True, max_samples=100)
            ```
        """
        # Convert normalized locations back to original pixel coordinates
        locations_pixel_coords = self.X_scaler.inverse_transform(locations)

        # Round locations to nearest integer and clip to valid dataset boundaries
        locations_pixel_coords = np.round(locations_pixel_coords).astype(int)
        locations_pixel_coords[:, 0] = np.clip(locations_pixel_coords[:, 0], 0,
                                               self.y.shape[0] - 1)
        locations_pixel_coords[:, 1] = np.clip(locations_pixel_coords[:, 1], 0,
                                               self.y.shape[1] - 1)

        # If continuous sensing is enabled, interpolate between points using skimage.draw.line
        if continuous_sening:
            interpolated_locs: List[np.ndarray] = []
            if locations_pixel_coords.shape[0] > 1:
                # Iterate through pairs of consecutive points to draw lines
                for i in range(locations_pixel_coords.shape[0] - 1):
                    loc1 = locations_pixel_coords[i]
                    loc2 = locations_pixel_coords[i + 1]
                    # line returns (row_coords, col_coords)
                    rr, cc = line(loc1[0], loc1[1], loc2[0], loc2[1])
                    interpolated_locs.append(np.column_stack((rr, cc)))

            # If there's only one point, or if no lines were drawn (e.g., due to identical consecutive points),
            # still include the initial locations.
            if not interpolated_locs:
                # If continuous sensing is true but no path, just return the initial locations if any
                if locations_pixel_coords.shape[0] > 0:
                    locations_pixel_coords = locations_pixel_coords
                else:
                    return np.empty((0, 2)), np.empty((0, 1))
            else:
                locations_pixel_coords = np.concatenate(interpolated_locs,
                                                        axis=0)

        # Ensure that locations_pixel_coords is not empty before indexing
        if locations_pixel_coords.shape[0] == 0:
            return np.empty((0, 2)), np.empty((0, 1))

        # Ensure indices are within bounds (should be handled by clip, but double check)
        valid_rows = np.clip(locations_pixel_coords[:, 0], 0,
                             self.y.shape[0] - 1)
        valid_cols = np.clip(locations_pixel_coords[:, 1], 0,
                             self.y.shape[1] - 1)

        # Extract data at the specified pixel locations
        data = self.y[valid_rows, valid_cols].reshape(-1, 1)

        # Drop NaN values from data and corresponding locations
        valid_mask = np.isfinite(data.ravel())
        locations_pixel_coords = locations_pixel_coords[valid_mask]
        data = data[valid_mask]

        # Re-normalize valid locations
        if locations_pixel_coords.shape[0] == 0:
            return np.empty((0, 2)), np.empty((0, 1))
        locations_normalized = self.X_scaler.transform(locations_pixel_coords)

        # Limit the number of samples to max_samples if needed
        if len(locations_normalized) > max_samples:
            indices = np.random.choice(len(locations_normalized),
                                       max_samples,
                                       replace=False)
            locations_normalized = locations_normalized[indices]
            data = data[indices]

        return locations_normalized.astype(self.dtype), data.astype(self.dtype)
