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

try:
    from osgeo import gdal
except:
    pass

####################################################
# Utils used to prepare synthetic datasets  

'''
Remove points inside polygons
'''
def remove_polygons(X, Y, polygons):
    points = np.array([X.flatten(), Y.flatten()]).T
    for polygon in polygons:
        p = path.Path(polygon)
        points = points[~p.contains_points(points)]
    return points[:, 0], points[:, 1]

'''
Remove points inside circle patches
'''
def remove_circle_patches(X, Y, circle_patches):
    points = np.array([X.flatten(), Y.flatten()]).T
    for circle_patch in circle_patches:
        points = points[~circle_patch.contains_points(points)]
    return points[:, 0], points[:, 1]

'''
Generate a point at a distance d from a point at angle theta

Args:
    point: (N, 2) array of points
    d: distance
    theta: angle in radians
'''
def point_pos(point, d, theta):
    return np.c_[point[:, 0] + d*np.cos(theta), point[:, 1] + d*np.sin(theta)]

####################################################

def prep_tif_dataset(dataset_path=None):
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

def get_dataset(dataset, dataset_path=None,
                num_train=1000,
                num_test=2500, 
                num_candidates=150):
    
    # Load the data
    if dataset == 'tif':
        X, y = prep_tif_dataset(dataset_path=dataset_path)
    elif dataset == 'synthetic':
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
