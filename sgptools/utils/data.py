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

####################################################
# Utils used to prepare synthetic datasets  

def remove_polygons(X, Y, polygons):
    '''
    Remove points inside polygons.

    Args:
        X  (ndarray): (N,); array of x-coordinate
        Y  (ndarray): (N,); array of y-coordinate
        polygons (list of matplotlib path polygon): Polygons to remove from the X, Y points

    Returns:
        X  (ndarray): (N,); array of x-coordinate
        Y  (ndarray): (N,); array of y-coordinate
    '''
    points = np.array([X.flatten(), Y.flatten()]).T
    for polygon in polygons:
        p = path.Path(polygon)
        points = points[~p.contains_points(points)]
    return points[:, 0], points[:, 1]

def remove_circle_patches(X, Y, circle_patches):
    '''
    Remove points inside polycircle patchesgons.

    Args:
        X  (ndarray): (N,); array of x-coordinate
        Y  (ndarray): (N,); array of y-coordinate
        circle_patches (list of matplotlib circle patches): Circle patches to remove from the X, Y points

    Returns:
        X  (ndarray): (N,); array of x-coordinate
        Y  (ndarray): (N,); array of y-coordinate
    '''
    points = np.array([X.flatten(), Y.flatten()]).T
    for circle_patch in circle_patches:
        points = points[~circle_patch.contains_points(points)]
    return points[:, 0], points[:, 1]

def point_pos(point, d, theta):
    '''
    Generate a point at a distance d from a point at angle theta.

    Args:
        point (ndarray): (N, 2); array of points
        d (float): distance
        theta (float): angle in radians

    Returns:
        X  (ndarray): (N,); array of x-coordinate
        Y  (ndarray): (N,); array of y-coordinate
    '''
    return np.c_[point[:, 0] + d*np.cos(theta), point[:, 1] + d*np.sin(theta)]

####################################################

def prep_tif_dataset(dataset_path, 
                     dim_max=2500):
    '''Load and preprocess a dataset from a GeoTIFF file (.tif file).

    Large tif files 
    need to be downsampled using the following terminal command: 
    ```gdalwarp -tr 50 50 <input>.tif <output>.tif```

    Args:
        dataset_path (str): Path to the dataset file, used only when dataset_type is 'tif'.
        dim_max (int): Maximum dimension of the dataset. If the dataset exceeds this size, it will be downsampled.

    Returns:
       y (ndarray): (n, 1); Dataset labels
    '''
    data = PIL.Image.open(dataset_path)
    data = np.array(data)
    print(f"Loaded dataset from {dataset_path} with shape {data.shape}")
    downsample = np.ceil(np.max(data.shape) / dim_max).astype(int)
    if downsample <= 1:
        downsample = 1
    else:
        print(f'Downsampling by a factor of {downsample} to fit the maximum dimension of {dim_max}.')
    data = data[::downsample, ::downsample].astype(float)
    data[np.where(data==-999999.0)] = np.nan
    return data

def prep_synthetic_dataset(shape=(1000, 1000), 
                           min_height=0.0, 
                           max_height=30.0, 
                           roughness=0.5,
                           random_seed=None,
                           **kwargs):
    '''Generates a 50x50 grid of synthetic elevation data using the diamond square algorithm.
    
    Refer to the following repo for more details:
        - [https://github.com/buckinha/DiamondSquare](https://github.com/buckinha/DiamondSquare)
    
    Args:
        shape (tuple): (x, y); Grid size along the x and y axis
        min_height (float): Minimum allowed height in the sampled data
        max_height (float): Maximum allowed height in the sampled data
        roughness (float): Roughness of the sampled data
        random_seed (int): Random seed for reproducibility

    Returns:
       X (ndarray): (n, d); Dataset input features
       y (ndarray): (n, 1); Dataset labels
    '''
    data = diamond_square(shape=shape,
                          min_height=min_height, 
                          max_height=max_height, 
                          roughness=roughness,
                          random_seed=random_seed,
                          **kwargs)
    return data.astype(float)

class Dataset:
    def __init__(self, 
                 dataset_path=None,
                 num_train=1000, 
                 num_test=2500, 
                 num_candidates=150,
                 **kwargs):
        """Class to load and preprocess a dataset. 

        This includes:
        - loading the dataset from a tif file or generating a synthetic dataset
        - Sampling training, testing, and candidate points
        - Standardizing the dataset

        The class also provides methods to retrieve the training, testing, and candidate points,
        as well as to get sensor data at specified locations or along a path.

        The dataset can be either a GeoTIFF file or a synthetic dataset generated using the diamond square algorithm.
        The dataset is expected to be a 2D array where each element represents a label (e.g., elevation, temperature, etc.).
        The training, testing, and candidate points are sampled from the valid points in the dataset, i.e., points where the label is not NaN.

        Args:
            dataset_path (str): Path to the dataset file, used only when dataset_type is 'tif'.
            num_train (int): Number of training points to sample from the dataset.
            num_test (int): Number of testing points to sample from the dataset.
            num_candidates (int): Number of candidate points to sample from the dataset.
            **kwargs: Additional keyword arguments for synthetic dataset preparation (hkb_diamondsquare.DiamondSquare.diamond_square).
        """
        
        # Load/Create the data
        if dataset_path is not None:
            self.y = prep_tif_dataset(dataset_path=dataset_path, **kwargs)
        else:
            self.y = prep_synthetic_dataset(**kwargs)

        w, h = self.y.shape[0], self.y.shape[1]

        ########################################################
        
        # Create image coordinates array
        xx, yy = np.meshgrid(
            np.arange(self.y.shape[0]), 
            np.arange(self.y.shape[1])
        )
        X = np.stack((xx, yy), axis=-1).astype(int)
        print(f"Dataset shape: {self.y.shape}")

        ########################################################

        # Get valid points, i.e., points where the label is not NaN
        mask = np.where(np.isfinite(self.y))
        X_valid = np.column_stack((mask[0], mask[1]))

        # Get training points
        X_train = get_inducing_pts(X_valid, num_train, random=True)
        y_train = self.y[X_train[:, 0], X_train[:, 1]].reshape(-1, 1)

        # Get testing points
        X_test = get_inducing_pts(X_valid, num_test, random=True)
        y_test = self.y[X_test[:, 0], X_test[:, 1]].reshape(-1, 1)

        # Get candidate points
        X_candidates = get_inducing_pts(X_valid, num_candidates, random=True)

        ########################################################

        # Standardize dataset X coordinates
        self.X_scaler = StandardScaler()
        self.X_scaler.fit(X_train)

        # Change variance/scale parameter to ensure all axis are scaled to the same value
        # Additionally, scale the data to have an extent of at least 10.0 in each dimension
        ind = np.argmax(self.X_scaler.var_)
        self.X_scaler.var_ = np.ones_like(self.X_scaler.var_)*self.X_scaler.var_[ind]
        self.X_scaler.scale_ = np.ones_like(self.X_scaler.scale_)*self.X_scaler.scale_[ind]
        self.X_scaler.scale_ /= 10.0

        self.X_train = self.X_scaler.transform(X_train)
        self.X_test = self.X_scaler.transform(X_test)
        self.candidates = self.X_scaler.transform(X_candidates)

        # Standardize dataset labels
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(y_train)

        self.y_train = self.y_scaler.transform(y_train)
        self.y_test = self.y_scaler.transform(y_test)
        self.y = self.y_scaler.transform(self.y.reshape(-1, 1)).reshape(w, h)

        print(f"Training data shape: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}, {self.y_test.shape}")
        print(f"Candidate data shape: {self.candidates.shape}")
        print(f"Dataset loaded and preprocessed successfully.")

    def get_train(self):
        return self.X_train, self.y_train
    
    def get_test(self):
        return self.X_test, self.y_test
    
    def get_candidates(self):
        return self.candidates
        
    def get_sensor_data(self, locations, 
                        continuous_sening=False,
                        max_samples=500):
        """
        Method to get sensor data at specified locations or along a path.
        Args:
            locations (ndarray): (n, 2); Array of locations where sensor data is to be sampled.
                                 Each location is specified as normalized (x, y) coordinates.
            continuous_sening (bool): If True, interpolate between points to simulate continuous sensing.
            max_samples (int): Maximum number of samples to return.
        Returns:
            locations (ndarray): (m, 2); Normalized locations where sensor data was sampled.
                                 Each location is specified as normalized (x, y) coordinates.
            data (ndarray): (m, 1); Sensor data sampled at the specified locations.
        """
        # Convert normalized locations back to original pixel coordinates
        locations = self.X_scaler.inverse_transform(locations)

        # Round locations to nearest integer and clip to valid range
        locations = np.round(locations).astype(int)
        locations[:, 0] = np.clip(locations[:, 0], 0, self.y.shape[0] - 1)
        locations[:, 1] = np.clip(locations[:, 1], 0, self.y.shape[1] - 1)

        # If continuous sensing is enabled, interpolate between points
        if continuous_sening:
            locs = []
            for loc1, loc2 in zip(locations[1:], locations[:-1]):
                rr, cc = line(loc1[1], loc1[0], loc2[1], loc2[0])
                locs.append(np.column_stack((cc, rr)))
            locations = np.concatenate(locs, axis=0)

        # Extract data at the specified locations
        data = self.y[locations[:, 0], locations[:, 1]].reshape(-1, 1)
        
        # Drop NaN values from data
        valid_mask = np.isfinite(data.ravel())
        locations = locations[valid_mask]
        data = data[valid_mask]

        # Re-normalize locations
        if len(locations) == 0:
            return np.empty((0, 2)), np.empty((0, 1))
        locations = self.X_scaler.transform(locations)

        # Limit the number of samples to max_samples
        if len(locations) > max_samples:
            indices = np.random.choice(len(locations), max_samples, replace=False)
            locations = locations[indices]
            data = data[indices]
        else:
            indices = np.arange(len(locations))

        return locations, data