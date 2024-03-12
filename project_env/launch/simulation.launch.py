import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro
import yaml

def generate_launch_description():
    world_path = os.path.join(
        get_package_share_directory('project_env'),
        'worlds',
        'simulation.world')
    
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'),'launch'),'/gazebo.launch.py']),
                launch_arguments={'world':world_path}.items(),
    )

    simulation_description_path = os.path.join(get_package_share_directory('project_env'))
    simulation_urdf_path = os.path.join(simulation_description_path,'urdf','camera.urdf')
    robot_description_config = open(simulation_urdf_path).read()
    robot_description = {'robot_description' : robot_description_config}
    spawn_entity = Node(package='gazebo_ros', executable="spawn_entity.py",
                        arguments=['-file',simulation_urdf_path,
                                    '-entity','camera',
                                    '-x','-0.5',
                                    '-z','1',
                                    '-P','0.785'],
                        output='both' )

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    nodes = [
        gazebo,
        spawn_entity,
        node_robot_state_publisher
    ]

    return LaunchDescription(nodes)
