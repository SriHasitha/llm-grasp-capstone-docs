import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ros2_data.action import MoveXYZW
from geometry_msgs.msg import PoseStamped

class MoveXYZWActionClient(Node):

    def __init__(self):
        super().__init__('move_xyzw_action_client')
        self.action_client = ActionClient(self, MoveXYZW, '/MoveXYZW')

    def send_goal(self, pose):
        self.get_logger().info('Sending goal...')
        goal = MoveXYZW.Goal()
        goal.positionx = pose['positionx']
        goal.positiony = pose['positiony']
        goal.positionz = pose['positionz']
        goal.yaw = pose['yaw']
        goal.pitch = pose['pitch']
        goal.roll = pose['roll']
        goal.speed = pose['speed']

        self.action_client.wait_for_server()

        self.send_goal_future = self.action_client.send_goal_async(goal)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {}'.format(result.result))
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    client = MoveXYZWActionClient()

    pose_ready = {'positionx': 0.70, 'positiony': 0.30, 'positionz': 1.35,
                    'yaw': -45.0, 'pitch': 45.0, 'roll': 180.0, 'speed': 1.0}
    
    pose_spawn = {'positionx': 0.00, 'positiony': 0.00, 'positionz': 1.12,
                    'yaw': -90.0, 'pitch': -45.0, 'roll': -90.0, 'speed': 1.0}
    
    client.send_goal(pose_ready)

    rclpy.spin(client)

if __name__ == '__main__':
    main()
