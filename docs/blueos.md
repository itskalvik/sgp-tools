# [BlueOS SGP-Tools :simple-github:](https://github.com/itskalvik/docker-sgp-tools/tree/main/robot-docker)
#### A [BlueOS](https://blueos.cloud/) Extension for Autonomous Approximate Bathymetric Surveys

<div style="text-align:left">
<img height="60" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/blueos_sgptools.png">
</div>

## What Does It Do?
Autonomous Surface Vehicles (ASVs), such as the [BlueRobotics BlueBoat](https://bluerobotics.com/store/boat/blueboat/blueboat/), are well-suited for bathymetric surveys. However, it is often the case that an exhaustive survey mapping the depth at every location in an area is infeasible or unnecessary. In such cases, we can leverage variations in the underwater terrain to determine a few critical locations for data collection, which would result in a good approximation of the area's bathymetry.

The [SGP-Tools python library](https://www.itskalvik.com/sgp-tools) provides path planners to address the above problem, known as the informative path planning problem. The BlueOS SGP-Tools extension uses this library to determine ideal locations for the ASV to collect data and controls the ASV to autonomously visit the selected locations.

The following shows the path planner adaptively planning a path for an aerial drone with a downward-facing camera tasked with surveying a given area:
<div style="text-align:left">
<p><a href="http://itskalvik.com/sgp-tools">
<img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/AIPP-non-point_sensing.gif">
</a></p>
</div>


## Usage

### Starting a Mission
1. First, we need to define the survey area and the robot launch location. The extension can read this data from [QGC plan files](https://docs.qgroundcontrol.com/Stable_V4.3/en/qgc-user-guide/plan_view/plan_geofence.html). The survey area and launch position must be defined using a **polygon-shaped** geofence drawn in [QGC](https://qgroundcontrol.com/) and saved as ```mission.plan```.

    <div style="text-align:left">
    <img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/generate_plan.gif">
    </a></p>
    </div>

2. Once you have the plan file, copy it to the robot using the ```File Browser``` feature in BlueOS's [```Pirate Mode```](https://blueos.cloud/docs/1.0/usage/advanced/). The ```mission.plan``` file should be uploaded to the following directory: ```/extensions/sgptools/```

    Once the ```mission.plan``` file is uploaded, restart the extension to ensure that the new file is used for the mission.

    <div style="text-align:left">
    <img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/upload_mission.gif">
    </a></p>
    </div>

3. Finally, use the terminal provided by the SGP-Tools extension to start the mission with the following command:
    ```
    ros2 launch ros_sgp_tools single_robot.launch.py
    ```

    <div style="text-align:left">
    <img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/start_mission.gif">
    </a></p>
    </div>

### Viewing the Data after a Mission
The sensor data, along with the corresponding GPS coordinates, will be logged to an [HDF5](https://docs.h5py.org/en/stable/) file in the ```DATA_FOLDER```, where the ```mission.plan``` was uploaded. 

We can estimate the bathymetry of the entire survey area using the collected data and visualize a normalized version with the following command:

⚠️ Do not run this during the mission, as it will disrupt the path planner
```
ros2 launch ros_sgp_tools visualize_data.launch.py
```

The above command will publish a point cloud that can be viewed using [foxglove](https://foxglove.dev/product). You can access it from a web browser at [https://app.foxglove.dev/](https://app.foxglove.dev/). Use the ```open connection``` feature and change the address from ```localhost``` to the IP address of the ASV.

<div style="text-align:left">
<img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/data_viz.gif">
</a></p>
</div>

You can control the point cloud density using the ```num_samples``` parameter. You can set this from foxglove's ```Parameters``` panel.

By default, the latest mission log will be visualized. You can visualize a specific mission log using the following command (replace ```<log folder name>``` with  the log folder name):

```
ros2 launch ros_sgp_tools visualize_data.launch.py mission_log:=<log folder name>
```

## Parameters
You can control the following extension parameters by running the following command in the terminal provided by the SGP-Tools extension:

```
export <parameter_name>=<parameter_value>
```

The parameters reset to their default values after rebooting. They can be made permanent by configuring the parameters using the app environment variables on the BlueOS extensions page in pirate mode.

### Available Parameters: 

* ```PING2_PORT``` (```default: /dev/ttyUSB0```):
    - Specifies the device to which the Ping2 sonar is mounted. You can get the device port from the ```Ping Sonar Devices``` page in BlueOS.

* ```NUM_WAYPOINTS``` (```default: 20```):
    - The number of waypoints optimized by the path planner.
    - Increasing the number of waypoints results in a more complex path that covers a larger area.
    - Recommend increasing only when the default setting results in a poor reconstruction of the environment, as this increases the computational cost and leads to slower online path updates.

* ```SAMPLING_RATE``` (```default: 2```): 
    - The number of points to sample along each edge of the path during path planning. The default value of ```2```  means that only the vertices of the path are used.
    - The path planner assumes the data is collected only at the sampled points. Increasing the sampling rate allows the planner to better approximate the information along the entire path, thereby resulting in more informative paths.
    - Recommend increasing only when the default setting results in a poor reconstruction of the environment, as this increases the computational cost and leads to slower online path updates.

* ```DATA_BUFFER_SIZE``` (```default: 100```):
    - The number of sensor data samples to collect before using the data to update the model parameters, which, in turn, will be used to update future waypoints.
    - Increasing the buffer size will allow the planner to compute better parameter estimates, which will result in more informative paths.
    - Recommend increasing only when the default setting results in a poor reconstruction of the environment, as this increases the computational cost and leads to slower online path updates.

* ```TRAIN_PARAM_INDUCING``` (```default: False```):
    - Enables training the inducing points of the parameter model (sparse Gaussian process) in addition to the kernel parameters during the path update.
    - Enabling this feature will result in more accurate parameter estimates and more informative paths.
    - Recommend increasing only when the default setting results in a poor reconstruction of the environment, as this increases the computational cost and leads to slower online path updates.

* ```NUM_PARAM_INDUCING``` (```default: 40```):
    - The number of inducing points used in the parameter model (sparse Gaussian process).
    - Increasing the number of inducing points will result in more accurate parameter estimates and more informative paths.
    - Recommend enabling only when the default setting results in a poor reconstruction of the environment, as this increases the computational cost and leads to slower online path updates.

* ```START_FOXGLOVE``` (```default: False```): 
    - Enables [foxglove](https://foxglove.dev/product), a web-based data visualization platform similar to [RViz](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html).
    - Currently it does not publish any new information that is not available through Cockpit or QGC.

* ```ADAPTIVE_IPP``` (```default: True```):
    - Enables adaptive informative path planning.
    - When enabled, it uses the data streaming from the sonar to learn the correlations in the underwater bathymetry and further optimizes the future waypoints to collect even more informative data.
    - Recommend disabling it if the onboard sonar is currently unsuppored by this package. 
    
* ```NAMESPACE``` (```default: robot_0```): 
    - ROS2 namespace, useful when multiple ROS2 robots are operating on the same network.
    - Currently, only the single robot planner is fully supported.
    
* ```DATA_TYPE``` (```default: Ping2```): 
    - Type of sensor to be used by the path planner. 
    - Currently, only the [BlueRobotics Ping Sonar](https://bluerobotics.com/store/sonars/echosounders/ping-sonar-r2-rp/) is supported.

* ```FAKE_DATA``` (```default: False```):
    - Enables a fake sensor data publisher used only for testing. Requires ```DATA_TYPE=SerialPing2```.

## Hardware Configuration
- This extension works only on 64-bit operating systems. You can install the latest version of [BlueOS](https://github.com/bluerobotics/BlueOS) on [64-bit Raspberry Pi OS Lite](https://www.raspberrypi.com/software/operating-systems/) by running the following command on the Pi (ensure the username is set to ```pi```):
    ```
    sudo su -c 'curl -fsSL https://raw.githubusercontent.com/bluerobotics/blueos-docker/master/install/install.sh | bash'
    ```

- The extension requires over 4GB of memory+swap. Please ensure that the swap size is large enough to accommodate the extension. The extension will copy the shell script ```config_swap.sh``` to ```/usr/blueos/extensions/sgptools/``` folder on the underlying device. You can use this script to increase the swap size before starting the path planner. 

    You will have to use [```Pirate Mode```](https://blueos.cloud/docs/1.0/usage/advanced/) to access BlueOS's built-in terminal and run the script on the underlying device via the ```red-pill``` utility. Use the following commands to enable ```red-pill``` and increase the swap size: 
    ```
    red-pill
    sudo bash /usr/blueos/extensions/sgptools/config_swap.sh
    ```

    <div style="text-align:left">
    <img width="472" src="https://raw.githubusercontent.com/itskalvik/docker-sgp-tools/refs/heads/main/.assets/upload_mission.gif">
    </a></p>
    </div>

## Disclaimer ⚠️
This extension, when executed properly, will take control of the ASV and could potentially collide the vehicle with obstacles in the environment. Please use it with caution.