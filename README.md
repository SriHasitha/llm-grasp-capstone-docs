# ChatterArm: Large Language Model Augmented Vision-based Grasping​
## RBE 594 Capstone Project
### Authors : Sri Lakshmi Hasitha Bachimanchi, Dheeraj Bhogisetty, Soham Shantanu Aserkar​

## Abstract

With an aging global population, the need for scalable and effective elderly care solutions is becoming increasingly urgent. This project addresses the challenges of providing support to the elderly in everyday tasks such as fetching objects. The approach combines multi-modal large language model (LLM) with vision-based grasping technique and a robot manipulator to create an interactive robot. Our system allows for interaction through natural language text input, enabling the robot to recognize and manipulate objects with variations in shape and color. Results from simulation tests show that the manipulator can successfully execute tasks based on user commands, demonstrating its potential to operate effectively in real-world scenarios. The impact of this technology extends beyond individual assistance, with potential applications in inventory management, order fulfilment, and waste sorting.

## Approach
We aim to integrate Language-Segment-Anything-Model(LangSAM) and Generative Grasp Convolutional Neural Network (GGCNN) into a ROS2 Service Client Framework.

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
## Running Integrated Pipeline of ROS2/Gazebo + GGCNN + LangSAM for inference

### Step 1: Run this service first
```sh
ros2 run langsam_vision vision_node
```
### Step 2: Launch gazebo environment
```sh
ros2 launch frankaproject_env panda_interface.launch.py
```
### Step 3: Initialize the GGCNN Service
```sh
ros2 run ros2_ggcnn ggcnn_service
```
### Step 4: Launch unified launch file to load all remaining services and send prompt to LangSAM to trigger prediction
```sh
ros2 run move_panda move_panda_client --ros-args -p prompt:'Pick up the green ball'
```
