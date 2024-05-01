#!/usr/bin/python3

# ===================================== COPYRIGHT ===================================== #
#                                                                                       #
#  IFRA (Intelligent Flexible Robotics and Assembly) Group, CRANFIELD UNIVERSITY        #
#  Created on behalf of the IFRA Group at Cranfield University, United Kingdom          #
#  E-mail: IFRA@cranfield.ac.uk                                                         #
#                                                                                       #
#  Licensed under the Apache-2.0 License.                                               #
#  You may not use this file except in compliance with the License.                     #
#  You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0  #
#                                                                                       #
#  Unless required by applicable law or agreed to in writing, software distributed      #
#  under the License is distributed on an "as-is" basis, without warranties or          #
#  conditions of any kind, either express or implied. See the License for the specific  #
#  language governing permissions and limitations under the License.                    #
#                                                                                       #
#  IFRA Group - Cranfield University                                                    #
#  AUTHORS: Mikel Bueno Viso - Mikel.Bueno-Viso@cranfield.ac.uk                         #
#           Seemal Asif      - s.asif@cranfield.ac.uk                                   #
#           Phil Webb        - p.f.webb@cranfield.ac.uk                                 #
#                                                                                       #
#  Date: September, 2022.                                                               #
#                                                                                       #
# ===================================== COPYRIGHT ===================================== #

# ======= CITE OUR WORK ======= #
# You can cite our work with the following statement:
# IFRA (2022) ROS2.0 ROBOT SIMULATION. URL: https://github.com/IFRA-Cranfield/ros2_RobotSimulation.

# panda.launch.py:
# Launch file for the PANDA ROBOT GAZEBO + MoveIt!2 SIMULATION in ROS2 Humble:

# Import libraries:
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro
import yaml

# LOAD FILE:
def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        # parent of IOError, OSError *and* WindowsError where available.
        return None
# LOAD YAML:
def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        # parent of IOError, OSError *and* WindowsError where available.
        return None

# ========== **GENERATE LAUNCH DESCRIPTION** ========== #
def generate_launch_description():

    # joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    # joint_positions = [0.0, 0.0, 0.0, -2.0, 0.0, 0.436, 0.0]


    # *********************** Gazebo *********************** # 
    
    # DECLARE Gazebo WORLD file:
    franka_env = os.path.join(
        get_package_share_directory('franka_env'),
        'worlds',
        'panda.world')
    # DECLARE Gazebo LAUNCH file:
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
                launch_arguments={'world': franka_env}.items(),
             )

    # ========== COMMAND LINE ARGUMENTS ========== #
    # print("")
    # print(" --- Cranfield University --- ")
    # print("        (c) IFRA Group        ")
    # print("")

    # print("ros2_RobotSimulation --> PANDA ROBOT")
    # print("Launch file -> panda_interface.launch.py")

    # print("")
    # print("Robot configuration:")
    # print("")

    # # Cell Layout:
    # print("- Cell layout:")
    # error = True
    # while (error == True):
    #     print("     + Option N1: PANDA ROBOT alone.")
    #     print("     + Option N2: PANDA ROBOT on top of a pedestal.")
    #     cell_layout = input ("  Please select: ")
    #     if (cell_layout == "1"):
    #         error = False
    #         cell_layout_1 = "true"
    #         cell_layout_2 = "false"
    #     elif (cell_layout == "2"):
    #         error = False
    #         cell_layout_1 = "false"
    #         cell_layout_2 = "true"
    #     else:
    #         print ("  Please select a valid option!")

    # Panda always on pedestal
    cell_layout_1 = "false"
    cell_layout_2 = "true"

    print("Launching Panda Gazebo Sim ...")
    print("")

    # End-Effector:
    # print("- End-effector:")
    # print("     + No EE variants for this robot.")
    EE_no = "true"
    
    # error = True
    # while (error == True):
    #     print("     + Option N1: No end-effector.")
    #     print("     + Option N2: ***.")
    #     end_effector = input ("  Please select: ")
    #     if (end_effector == "1"):
    #         error = False
    #         EE_no = "true"
    #         EE_*** = "false"
    #     elif (end_effector == "2"):
    #         error = False
    #         EE_no = "false"
    #         EE_*** = "true"
    #     else:
    #         print ("  Please select a valid option!")
    # print("")

    # ***** ROBOT DESCRIPTION ***** #
    # PANDA ROBOT Description file package:
    panda_description_path = os.path.join(
        get_package_share_directory('franka_env'))
    # FANUC LBR-panda ROBOT urdf file path:
    xacro_file = os.path.join(panda_description_path,
                              'urdf',
                              'panda.urdf.xacro')
    # Generate ROBOT_DESCRIPTION for PANDA ROBOT:
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc, mappings={
        "cell_layout_1": cell_layout_1,
        "cell_layout_2": cell_layout_2,
        "EE_no": EE_no,
        # "EE_**": EE_**,
        })
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}

    # SPAWN ROBOT TO GAZEBO:
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'panda'],
                        output='screen'#[{'joint_names': joint_names, 'joint_positions': joint_positions}]
    )

    # ***** STATIC TRANSFORM ***** #
    # NODE -> Static TF:
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )
    # Publish TF:
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[
            robot_description,
            {"use_sim_time": True}
        ]
    )

    # ***** ROS2_CONTROL -> LOAD CONTROLLERS ***** #

    # Joint STATE BROADCASTER:
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    # Joint TRAJECTORY Controller:
    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )
    
    # === Panda HAND === #
    panda_handleft_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_handleft_controller", "-c", "/controller_manager"],
    )
    panda_handright_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_handright_controller", "-c", "/controller_manager"],
    )


    # *********************** MoveIt!2 *********************** #   
    
    # Command-line argument: RVIZ file?
    rviz_arg = DeclareLaunchArgument(
        "rviz_file", default_value="False", description="Load RVIZ file."
    )

    # *** PLANNING CONTEXT *** #
    # Robot description, SRDF:
    robot_description_semantic_config = load_file("frankaproject_env", "config/panda.srdf")
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_config }
    
    # Kinematics.yaml file:
    kinematics_yaml = load_yaml("frankaproject_env", "config/kinematics.yaml")
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # Move group: OMPL Planning.
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_yaml = load_yaml("frankaproject_env", "config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # MoveIt!2 Controllers:
    moveit_simple_controllers_yaml = load_yaml("frankaproject_env", "config/panda_controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # START NODE -> MOVE GROUP:
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": True}, 
        ],
    )

    # RVIZ:
    # load_RVIZfile = LaunchConfiguration("rviz_file")
    # rviz_base = os.path.join(get_package_share_directory("frankaproject_env"), "config")
    # rviz_full_config = os.path.join(rviz_base, "panda_moveit2.rviz")
    # rviz_node_full = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="log",
    #     arguments=["-d", rviz_full_config],
    #     parameters=[
    #         robot_description,
    #         robot_description_semantic,
    #         ompl_planning_pipeline_config,
    #         kinematics_yaml,
    #         {"use_sim_time": True}, 
    #     ],
    #     condition=UnlessCondition(load_RVIZfile),
    # )

    # *********************** ROS2.0 Robot/End-Effector Actions/Triggers *********************** #
    # MoveJ ACTION:
    moveJ_interface = Node(
        name="moveJs_action",
        package="ros2_actions",
        executable="moveJs_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )

    # To spawn Panda in Ready2 pose: 
    # ros2 action send_goal -f /MoveJs ros2_data/action/MoveJs "{goal: {joint1: 0.00, joint2: 0.00, joint3: 0.00, joint4: 0.00, joint5: 0.00, joint6: 0.00, joint7: 0.00}, speed: 1.0}" # (7-DOF)
    spawn_panda = ExecuteProcess(
        cmd=[[
            'ros2 action send_goal -f ',
            # turtlesim_ns,
            '/MoveJs ',
            'ros2_data/action/MoveJs ',
            '"{goal: {joint1: 0.00, joint2: 0.00, joint3: 0.00, joint4: -114.592, joint5: 0.00, joint6: 24.98, joint7: 0.00}, speed: 1.0}"'
        ]],
        shell=True
    )

    # MoveG ACTION:
    moveG_interface = Node(
        name="moveG_action",
        package="ros2_actions",
        executable="moveG_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_gripper'}],
    )
    # MoveXYZW ACTION:
    moveXYZW_interface = Node(
        name="moveXYZW_action",
        package="ros2_actions",
        executable="moveXYZW_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveL ACTION:
    moveL_interface = Node(
        name="moveL_action",
        package="ros2_actions",
        executable="moveL_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveR ACTION:
    moveR_interface = Node(
        name="moveRs_action",
        package="ros2_actions",
        executable="moveRs_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveXYZ ACTION:
    moveXYZ_interface = Node(
        name="moveXYZ_action",
        package="ros2_actions",
        executable="moveXYZ_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveYPR ACTION:
    moveYPR_interface = Node(
        name="moveYPR_action",
        package="ros2_actions",
        executable="moveYPR_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveROT ACTION:
    moveROT_interface = Node(
        name="moveROT_action",
        package="ros2_actions",
        executable="moveROT_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )
    # MoveRP ACTION:
    moveRP_interface = Node(
        name="moveRP_action",
        package="ros2_actions",
        executable="moveRP_action",
        output="screen",
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {"use_sim_time": True}, {"ROB_PARAM": 'panda_arm'}],
    )

    # ATTACHER action for ros2_grasping plugin:
    Attacher = Node(
        name="ATTACHER_action",
        package="ros2_grasping",
        executable="attacher_action.py",
        output="screen",
    )

    # LangSAM node:
    LangSAM_node = Node(
        name="langsam_service_node",
        package="langsam_vision",
        executable="vision_node",
        output="screen",
    )

    # GGCNN node:
    GGCNN_node = Node(
        name="ggcnn_service_node",
        package="ros2_ggcnn",
        executable="ggcnn_service",
        output="screen",
    )
    
    return LaunchDescription(
        [
            # LangSAM node:
            # LangSAM_node,
            # GGCNN node:
            # GGCNN_node,
            # Gazebo nodes:
            gazebo, 
            spawn_panda,
            spawn_entity,
            # ROS2_CONTROL:
            static_tf,
            robot_state_publisher,
            
            # ROS2 Controllers:
            RegisterEventHandler(
                OnProcessExit(
                    target_action = spawn_entity,
                    on_exit = [
                        joint_state_broadcaster_spawner,
                    ]
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action = joint_state_broadcaster_spawner,
                    on_exit = [
                        joint_trajectory_controller_spawner,
                    ]
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action = joint_trajectory_controller_spawner,
                    on_exit = [
                        panda_handleft_controller_spawner,
                    ]
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action = panda_handleft_controller_spawner,
                    on_exit = [
                        panda_handright_controller_spawner,
                    ]
                )
            ),

            RegisterEventHandler(
                OnProcessExit(
                    target_action = panda_handright_controller_spawner,
                    on_exit = [

                        # MoveIt!2:
                        TimerAction(
                            period=5.0,
                            actions=[
                                rviz_arg,
                                # rviz_node_full,
                                run_move_group_node
                            ]
                        ),

                        # ROS2.0 Actions:
                        TimerAction(
                            period=2.0,
                            actions=[
                                moveJ_interface,
                                # spawn_panda,
                                moveG_interface,
                                moveL_interface,
                                moveR_interface,
                                moveXYZ_interface,
                                moveXYZW_interface,
                                moveYPR_interface,
                                moveROT_interface,
                                moveRP_interface,
                                Attacher,
                            ]
                        ),

                    ]
                )
            )
        ]
    )