![Image title](assets/SGP-Tools.png#only-light)
![Image title](assets/logo_dark.png#only-dark)


# [ros_sgp_tools :simple-github:](https://github.com/itskalvik/ros_sgp_tools)
The [ros_sgp_tools](https://github.com/itskalvik/ros_sgp_tools) package provides a [ROS2](https://github.com/ros2) companion package for the SGP-Tools python library that can be deployed on [ArduPilot-based vehicles](https://ardupilot.org/copter/docs/common-use-cases-and-applications.html). 

- The package can be used to run online/adaptive IPP on ArduPilot based UGVs and ASVs. 
- The package can also be used with Gazebo/Ardupilot SITL.
- To use our Docker container with the preconfigured development environment, please refer to the documentation [here](docker-sgp-tools.html). 

### Running SGP-Tools Online/Adaptive IPP with Gazebo/ArduRover Simulator

<div style="text-align:left">
<img width="1000" src="assets/ros_demo.png">
</a></p>
</div>

Run the following commands in separate terminals:

- Launch Gazebo with the [AION R1 UGV](https://github.com/ArduPilot/SITL_Models/blob/master/Gazebo/docs/AionR1.md):
    ```
    gz sim -v4 -r r1_rover_runway.sdf
    ```
    To simulate a BlueBoat refer to this [documentation](https://github.com/ArduPilot/SITL_Models/blob/master/Gazebo/docs/BlueBoat.md).

- Launch [ArduRover SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html):
    ```
    sim_vehicle.py -v Rover -f rover-skid --model JSON --add-param-file=$HOME/SITL_Models/Gazebo/config/r1_rover.param --console --map -N -l 35.30371178789218,-80.73099267294185,0.,0.
    ```
    Note: 
    - Restart sim_vechile.py if you get the following message: ```paramftp: bad count```
    - Ensure the MAV Console shows that the vehicle has a GPS lock before running the next command
    - Ensure the MAV Map shows the vehicle before running the next command

- Launch the SGP-Tools Online/Adaptive IPP method:
    ```
    ros2 launch ros_sgp_tools single_robot.launch.py
    ```

### Environment setup
- To use our Docker container with the preconfigured development environment, please refer to the documentation [here](docker-sgp-tools.html). 

Alternatively, please follow the following instructions to configure the development envirnoment on your local machine. 

- Install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html#install-ros-2-packages)
  ```
  sudo apt install ros-humble-desktop
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
- Create a ROS 2 workspace
  ```
  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws/
  colcon build --symlink-install
  echo "source $HOME/ros2_ws/install/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
- Install [Gazebo Garden](https://gazebosim.org/docs/garden/install_ubuntu)
- Install [ArduPilot SITL](https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux)
- Install [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo?tab=readme-ov-file#installation)
- Get [SITL_Models repo](https://github.com/ArduPilot/SITL_Models)
- Setup environment variables

  ```
  echo "export GZ_VERSION=garden" >> ~/.bashrc
  echo "export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}" >> ~/.bashrc
  echo "export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/
  Gazebo/worlds:$GZ_SIM_RESOURCE_PATH" >> ~/.bashrc
  ```

### Package setup

- Install apt dependencies 
  ```
  sudo apt install ros-humble-mavros* ros-$ROS_DISTRO-cv-bridge
  ```
- Install pip dependencies 
  ```
  pip3 install -r requirements.txt
  ```
- Clone repo and build package
  ```
  cd ~/ros2_ws/src
  git clone https://github.com/itskalvik/ros_sgp_tools.git
  cd ..
  colcon build --symlink-install
  ```