### Setup 

1. Create ROS2 workspace
   
```sh
mkdir -p ~/llm-grasping-panda/src
```

2. Clone repo
```sh
cd ~/llm-grasping-panda/src
git clone https://github.com/SriHasitha/llm-grasp-capstone-docs.git
git clone -b humble https://github.com/nilseuropa/realsense_ros_gazebo.git
```
or
```sh
cd ~/llm-grasping-panda/src
git clone git@github.com:SriHasitha/llm-grasp-capstone-docs.git
git clone -b humble https://github.com/nilseuropa/realsense_ros_gazebo.git
```


3. Build packages
```sh
cd ~/llm-grasping-panda
colcon build
```

4. Source packages
```sh
source install/setup.bash
```

### Launch Panda Gazebo Simulation Environment

1. Launch only Gazebo sim:

```sh
ros2 launch franka_env panda_simulation.launch.py
```

2. Launch Gazebo + MoveIt!2 Environment

```sh
ros2 launch frankaproject_env panda.launch.py
```

3. Launch Gazebo + MoveIt!2 Environment + ROS2 Robot Triggers/Actions

```sh
ros2 launch frankaproject_env panda_interface.launch.py
```

### Move Panda Arm

Action client sends desired end-effector pose as goal to the /MoveXYZW action

```sh
ros2 run move_panda move_panda_client
```
### Run GGCNN Service

Initialize the GGCNN Service

```sh
ros2 run ros2_ggcnn ggcnn_service
```

Call the GGCNN service to predict grasp pose

```sh
ros2 service call /grasp_prediction ggcnn_interface/srv/GraspPrediction
``` 