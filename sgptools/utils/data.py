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
from .misc import get_inducing_pts, cont2disc
from sklearn.preprocessing import StandardScaler
from hkb_diamondsquare.DiamondSquare import diamond_square

# Load optional dependency  
try:
    from osgeo import gdal
except:
    pass


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
        polygons (list of matplotlib circle patches): Circle patches to remove from the X, Y points

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

def prep_tif_dataset(dataset_path):
    '''Load and preprocess a dataset from a GeoTIFF file (.tif file). The input features 
    are set to the x and y pixel block coordinates and the labels are read from the file.
    The method also removes all invalid points.

    Large tif files 
    need to be downsampled using the following command: 
    ```gdalwarp -tr 50 50 <input>.tif <output>.tif```

    Args:
        dataset_path (str): Path to the dataset file, used only when dataset_type is 'tif'.

    Returns:
       X: (n, d); Dataset input features
       y: (n, 1); Dataset labels
    '''
    ds = gdal.Open(dataset_path)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows)
    data[np.where(data==-999999.0)] = np.nan

    x1, x2 = np.where(np.isfinite(data))
    X = np.vstack([x1, x2]).T
    y = data[x1, x2].reshape(-1, 1)

    return X.astype(float), y.astype(float)

####################################################

def prep_synthetic_dataset():
    '''Generates a 50x50 grid of synthetic elevation data using the diamond square algorithm.
    ```https://github.com/buckinha/DiamondSquare```
    
    Args:

    Returns:
       X: (n, d); Dataset input features
       y: (n, 1); Dataset labels
    '''
    data = diamond_square(shape=(50,50), 
                          min_height=0, 
                          max_height=30, 
                          roughness=0.5)

    # create x and y coordinates from the extent
    x_coords = np.arange(0, data.shape[0])/10
    y_coords = np.arange(0, data.shape[1])/10
    xx, yy = np.meshgrid(x_coords, y_coords)
    X = np.c_[xx.ravel(), yy.ravel()]
    y = data.ravel()
    y = y.reshape(-1, 1)

    return X.astype(float), y.astype(float)

####################################################

def get_dataset(dataset_type, dataset_path=None,
                num_train=1000,
                num_test=2500, 
                num_candidates=150):
    """Method to generate/load datasets and preprocess them for SP/IPP. The method uses kmeans to 
    generate train and test sets.
    
    Args:
        dataset_type (str): 'tif' or 'synthetic'. 'tif' will load and proprocess data from a GeoTIFF file. 
                        'synthetic' will use the diamond square algorithm to generate synthetic elevation data.
        dataset_path (str): Path to the dataset file, used only when dataset_type is 'tif'.
        num_train (int): Number of training samples to generate.
        num_test (int): Number of testing samples to generate.
        num_candidates (int): Number of candidate locations to generate.

    Returns:
       X_train (ndarray): (n, d); Training set inputs
       y_train (ndarray): (n, 1); Training set labels
       X_test (ndarray): (n, d); Testing set inputs
       y_test (ndarray): (n, 1); Testing set labels
       candidates (ndarray): (n, d); Candidate sensor placement locations
       X: (n, d); Full dataset inputs
       y: (n, 1); Full dataset labels
    """
    # Load the data
    if dataset_type == 'tif':
        X, y = prep_tif_dataset(dataset_path=dataset_path)
    elif dataset_type == 'synthetic':
        X, y = prep_synthetic_dataset()

    X_train = get_inducing_pts(X, num_train)
    X_train, y_train = cont2disc(X_train, X, y)

    X_test = get_inducing_pts(X, num_test)
    X_test, y_test = cont2disc(X_test, X, y)

    candidates = get_inducing_pts(X, num_candidates)
    candidates = cont2disc(candidates, X)

    # Standardize data
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)*10.0
    X_test = X_scaler.transform(X_test)*10.0
    X = X_scaler.transform(X)*10.0

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    y = y_scaler.transform(y)

    return X_train, y_train, X_test, y_test, candidates, X, y
