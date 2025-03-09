# [ros_sgp_tools :simple-github:](https://github.com/itskalvik/ros_sgp_tools)
The [ros_sgp_tools](https://github.com/itskalvik/ros_sgp_tools) package provides a [ROS2](https://github.com/ros2) companion package for the SGP-Tools python library that can be deployed on [ArduPilot-based vehicles](https://ardupilot.org/copter/docs/common-use-cases-and-applications.html). 

- The package can be used to run online/adaptive IPP on ArduPilot based UGVs and ASVs. 
- The package can also be used with Gazebo/Ardupilot SITL.
- To use our Docker container with the preconfigured development environment, please refer to the documentation [here](docker-sgp-tools.html). 

<div style="text-align:left">
<img height="60" src="assets/ros2_ardupilot.png">
</div>

## Package setup
  ```
  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws/src
  git clone https://github.com/itskalvik/ros_sgp_tools.git
  cd ros_sgp_tools
  python3 -m pip install -r requirements.txt
  cd ~/ros2_ws
  rosdep install --from-paths src --ignore-src -y
  colcon build --symlink-install
  source ~/ros2_ws/install/setup.bash
  ```

## Running SGP-Tools Online/Adaptive IPP with Gazebo/BlueBoat Simulator

![Image title](assets/ros_demo.png)

Run the following commands in separate terminals:

- Launch Gazebo with the [Blue Robotics BlueBoat ASV](https://bluerobotics.com/store/boat/blueboat/blueboat/):
    ```
    gz sim -v4 -r blueboat_waves.sdf
    ```

- Launch [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html):
    ```
    sim_vehicle.py -v Rover -f rover-skid --model JSON --console --map -N -L RATBeach
    ```
    Note: 
    - Ensure the MAV Console shows `AHRS` and `GPS` in green before running the next command
    - Ensure the MAV Map shows the vehicle before running the next command
    - Restart sim_vechile.py if you get the following message: ```paramftp: bad count```

- Launch the [SGP-Tools](http://itskalvik.com/sgp-tools) Online/Adaptive IPP method:
    ```
    ros2 launch ros_sgp_tools single_robot.launch.py
    ```

### Environment setup
- To use our Docker container with the preconfigured development environment, please refer to the documentation [here](https://github.com/itskalvik/docker-sgp-tools?tab=readme-ov-file#docker-sgp-tools). 

Alternatively, please install the following packages to configure the development envirnoment on your local machine:

- [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html)
- [Gazebo Garden](https://gazebosim.org/docs/garden/install_ubuntu)
- [ArduPilot SITL](https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux)
- [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo?tab=readme-ov-file#installation)
- [Wave Sim](https://github.com/srmainwaring/asv_wave_sim)
- [SITL_Models repo](https://github.com/ArduPilot/SITL_Models)