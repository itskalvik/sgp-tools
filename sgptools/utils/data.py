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
import pandas as pd
from matplotlib import path
from .misc import get_inducing_pts, cont2disc
from sklearn.preprocessing import StandardScaler
from hkb_diamondsquare.DiamondSquare import diamond_square

try:
    import netCDF4 as nc
except:
  pass

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 317500000


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

def get_intel_dataset(locs_file, data_file):
    # Get sensor data
    df = pd.read_csv(data_file, sep=' ', header=None, 
                     names=['date', 'time', 'epoch', 'modeid', 
                            'temperature', 'humidity', 'light', 'voltage'])
    df = df.drop(['humidity', 'light', 'voltage'], axis=1)

    # Remove entries with invalid temperatures
    df = df.dropna()
    df = df.drop(df[df.temperature < 12].index)
    df = df.drop(df[df.temperature > 38].index)

    # Remove invalid modes
    del_modes = [55, 56, 58, 6485, 33117, 65407]
    df = df[~df['modeid'].isin(del_modes)]

    valid_modes = np.sort(df.modeid.unique().astype(int))

    # Split dfs by modeid
    mode_dfs = []
    for id in np.sort(df.modeid.unique().astype(int)):
        tmp_df = df[df['modeid'] == id]
        tmp_df = tmp_df.drop(['modeid'], axis=1)
        tmp_df = tmp_df.rename(columns={'temperature': id})
        mode_dfs.append(tmp_df)

    # Merge dfs into single df with each mode's temp in seperate column
    df_sep = mode_dfs[0]    
    for df in mode_dfs[1:]:
        df_sep = pd.merge(df_sep, df, on=['epoch', 'date', 'time'], how='outer')
    df = df_sep

    df = df.sort_values(by=['date', 'time'])

    # Get sensor locs
    loc_df = pd.read_csv(locs_file, sep=' ', header=None, 
                        names=['modeid', 'x', 'y'])
    loc_df = loc_df[loc_df['modeid'].isin(valid_modes)][['x', 'y']].values

    # Remove invalid modes with missing data
    df = df.drop([15, 49, 27], axis=1)
    loc_df = np.delete(loc_df, [15, 49, 27], axis=0)

    return loc_df, df

def get_day_data(df, date_id, aggregation_rate=1000):
    # Get data for a single day
    dates = np.sort(df['date'].unique())
    day_df = df[df.date == dates[date_id]].drop(['date', 'time', 'epoch'], axis=1)
    
    # Take mean of each modeid's temperature over aggregation_rate (rows) intervals
    day_df = day_df.groupby(np.arange(len(day_df))//aggregation_rate).mean()
    
    # Clean data
    day_df = day_df.interpolate(method='cubic')
    day_df = day_df.dropna()
    return day_df.values.copy()  

def prep_intel_dataset(dataset_path=None):
    if dataset_path is None:
        x_path = 'datasets/intel-temperature-data/mote_locs.txt'
        y_path = 'datasets/intel-temperature-data/data.txt'
    candidates, df = get_intel_dataset(x_path, y_path)

    # Get sensor data from day 1 to learn the kernel parameters
    y_train = get_day_data(df, 0)
    X_train = np.expand_dims(candidates, 0).repeat(y_train.shape[0], 0)
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 2)

    # Get sensor data from day 2 to test
    y_test = get_day_data(df, 1)
    X_test = np.expand_dims(candidates, 0).repeat(y_test.shape[0], 0)
    y_test = y_test.reshape(-1, 1)
    X_test = X_test.reshape(-1, 2)

    return X_train, y_train, X_test, y_test, candidates

####################################################

def get_precipitation_dataset(filename, aggregation_rate=100):
    ds = nc.Dataset(filename)
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])
    X = np.array(np.meshgrid(lon, lat)).transpose(1, 2, 0)
    y = np.array(ds.variables['data'][:])

    X = X.reshape(-1, 2)
    y = y.reshape(len(y), -1)
    
    # Remove invalid modes
    valid_idx = np.where((y.std(axis=0) > 1) * (y.std(axis=0) < 100))
    X = X[valid_idx].reshape(-1, 2)
    y = y[:, valid_idx].reshape(len(y), -1, 1)

    X, y = X.astype(float), y.astype(float)

    # Take mean of each modeid's data over aggregation_rate (rows) intervals
    y = np.mean(y[:(len(y)//aggregation_rate)*aggregation_rate].reshape(-1,aggregation_rate,167,1), axis=1)

    return X, y

def prep_precip_dataset(num_train=15, num_test=10, dataset_path=None):
    if dataset_path is None:
        dataset_path='datasets/precipitation.nc'
    X, y = get_precipitation_dataset(dataset_path)

    # Get sensor data from num_train days to learn the kernel parameters
    y_train = y[:num_train]
    X_train = np.expand_dims(X, 0).repeat(y_train.shape[0], 0)
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 2)

    # Get sensor data from num_test days to test
    y_test = y[num_train:num_train+num_test]
    X_test = np.expand_dims(X, 0).repeat(y_test.shape[0], 0)
    y_test = y_test.reshape(-1, 1)
    X_test = X_test.reshape(-1, 2)

    candidates = X.copy()

    return X_train, y_train, X_test, y_test, candidates

####################################################

def get_salinity_dataset(filename, sample_rate=2):
    ds = nc.Dataset(filename)
    y = np.array(ds.variables['salt'])[0, :-1, ::sample_rate, ::sample_rate]

    # create x and y coordinates from the extent
    x_coords = np.arange(0, y.shape[2])/10
    y_coords = np.arange(0, y.shape[1])/10
    xx, yy = np.meshgrid(x_coords, y_coords)
    X = np.stack([xx, yy], axis=2)

    X = X.reshape(-1, 2)
    y = y.reshape(len(y), -1, 1)
    
    X, y = X.astype(float), y.astype(float)

    idx = np.where(~np.isnan(y[0]))[0]
    X = X[idx]
    y = y[:, idx]

    return X, y

def prep_salinity_dataset(num_train=3, sample_rate=2, dataset_path=None):
    if dataset_path is None:
        dataset_path='datasets/salinity.nc'
    X, y = get_salinity_dataset(dataset_path, sample_rate=sample_rate)

    # Get sensor data from num_train days to learn the kernel parameters
    y_train = y[:num_train]
    X_train = np.expand_dims(X, 0).repeat(y_train.shape[0], 0)
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 2)

    # Get sensor data from num_test days to test
    y_test = y[num_train:]
    X_test = np.expand_dims(X, 0).repeat(y_test.shape[0], 0)
    y_test = y_test.reshape(-1, 1)
    X_test = X_test.reshape(-1, 2)

    candidates = X.copy()

    return X_train, y_train, X_test, y_test, candidates

####################################################

def prep_soil_dataset(dataset_path=None, aggregation_rate=25):
    # Load data
    if dataset_path is None:
        dataset_path='datasets/soil.tif'
    data = PIL.Image.open(dataset_path)
    data = np.array(data)

    # create x and y coordinates from the extent
    x_coords = np.arange(0, data.shape[1])/10
    y_coords = np.arange(data.shape[0], 0, -1)/10
    xx, yy = np.meshgrid(x_coords, y_coords)
    X = np.c_[xx.ravel(), yy.ravel()]
    y = data.ravel()

    # Remove nans
    X = X[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # Aggregate data
    X = X[::aggregation_rate]
    y = y[::aggregation_rate]

    y = y.reshape(-1, 1)
    X = X.reshape(-1, 2)

    return X.astype(float), y.astype(float)

####################################################

def prep_elevation_dataset(dataset_path=None, extent=[500, 1000, 500, 1000]):
    # Load data
    if dataset_path is None:
        dataset_path='datasets/elevation.tif'
    data = PIL.Image.open(dataset_path)
    data = np.array(data)

    data = data[extent[0]:extent[1], extent[2]:extent[3]]

    # create x and y coordinates from the extent
    x_coords = np.arange(0, data.shape[0])/10
    y_coords = np.arange(0, data.shape[1])/10
    xx, yy = np.meshgrid(x_coords, y_coords)
    X = np.c_[xx.ravel(), yy.ravel()]
    y = data.ravel()

    # Remove background
    background_idx = np.where(y == np.max(y))[0]
    X = np.delete(X, background_idx, axis=0)
    y = np.delete(y, background_idx, axis=0)

    y = y.reshape(-1, 1)

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
    if dataset == 'intel':
        X_train, y_train, X_test, y_test, candidates = prep_intel_dataset(dataset_path=dataset_path)
        X = X_test
        y = y_test
    elif dataset == 'precipitation':
        X_train, y_train, X_test, y_test, candidates = prep_precip_dataset(num_train=15,
                                                                           dataset_path=dataset_path)
        X = X_test
        y = y_test
    elif dataset == 'salinity':
        X, y, _, _, _ = prep_salinity_dataset(sample_rate=1, num_train=1, 
                                              dataset_path=dataset_path)
    elif dataset == 'soil':
        X, y = prep_soil_dataset(dataset_path=dataset_path)
    elif dataset == 'elevation':
        X, y = prep_elevation_dataset(dataset_path=dataset_path)
    elif dataset == 'synthetic':
        X, y = prep_synthetic_dataset()

    if dataset in ['salinity', 'soil' ,'elevation', 'synthetic']:
        X_train = get_inducing_pts(X, num_train)
        X_train, y_train = cont2disc(X_train, X, y)

        X_test = get_inducing_pts(X, num_test)
        X_test, y_test = cont2disc(X_test, X, y)

        candidates = get_inducing_pts(X, num_candidates)
        candidates = cont2disc(candidates, X)

    # Standardize data
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    y = y_scaler.transform(y)

    return X_train, y_train, X_test, y_test, candidates, X, y
