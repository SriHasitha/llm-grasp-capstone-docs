import time
import rclpy
import sys
# from clients import *

from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from ros2_data.action import MoveXYZW, MoveG, MoveL, MoveXYZ
from ros2_grasping.action import Attacher

# Define GLOBAL VARIABLE -> RES:
RES = "null"

class MoveXYZWclient(Node):
    
    def __init__(self):
        # Initialise node:
        super().__init__('MoveXYZW_client')
        self._action_client = ActionClient(self, MoveXYZW, 'MoveXYZW')
        # Wait for MoveXYZW server to be available:
        print ("Waiting for MoveXYZW action server to be available...")
        self._action_client.wait_for_server()
        print ("MoveXYZW ACTION SERVER detected.")
    
    def send_goal(self, pose):
        self.get_logger().info('Sending pose goal...')
        goal_msg = MoveXYZW.Goal()
        goal_msg.positionx = pose['positionx']
        goal_msg.positiony = pose['positiony']
        goal_msg.positionz = pose['positionz']
        goal_msg.yaw = pose['yaw']
        goal_msg.pitch = pose['pitch']
        goal_msg.roll = pose['roll']
        goal_msg.speed = pose['speed']

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        global RES
        # Assign RESULT variable:
        result = future.result().result
        RES = result.result
        # Print RESULT:
        print ("MoveXYZW ACTION CALL finished.")     

    def feedback_callback(self, feedback_msg):
        # Assign FEEDBACK variable:
        feedback = feedback_msg.feedback
        # NO FEEDBACK NEEDED IN MoveXYZW ACTION CALL.

class MoveGclient(Node):
    
    def __init__(self):
        # Initialise node:
        super().__init__('MoveG_client')
        self._action_client = ActionClient(self, MoveG, 'MoveG')
        # Wait for MoveG server to be available:
        print ("Waiting for MoveG action server to be available...")
        self._action_client.wait_for_server()
        print ("MoveG ACTION SERVER detected.")
    
    def send_goal(self, GP):
        # Assign variables:
        goal_msg = MoveG.Goal()
        goal_msg.goal = GP['goal']
        # ACTION CALL:
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        global RES
        # Assign RESULT variable:
        result = future.result().result
        RES = result.result
        # Print RESULT:
        print ("MoveG ACTION CALL finished.")

    def feedback_callback(self, feedback_msg):
        # Assign FEEDBACK variable:
        feedback = feedback_msg.feedback
        # NO FEEDBACK NEEDED IN MoveG ACTION CALL.

class MoveLclient(Node):
    
    def __init__(self):
        # 1. Initialise node:
        super().__init__('MoveL_client')
        self._action_client = ActionClient(self, MoveL, 'MoveL')
        # 2. Wait for MoveL server to be available:
        print ("Waiting for MoveL action server to be available...")
        self._action_client.wait_for_server()
        print ("MoveL ACTION SERVER detected.")
    
    def send_goal(self, msg):
        # 1. Assign variables:
        goal_msg = MoveL.Goal()
        goal_msg.movex = msg['movex']
        goal_msg.movey = msg['movey']
        goal_msg.movez = msg['movez']
        goal_msg.speed = msg['speed']
        # 2. ACTION CALL:
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        global RES
        # 1. Assign RESULT variable:
        result = future.result().result
        RES = result.result
        # 2. Print RESULT:
        print ("MoveL ACTION CALL finished.")       

    def feedback_callback(self, feedback_msg):
        # 1. Assign FEEDBACK variable:
        feedback = feedback_msg.feedback
        # NO FEEDBACK NEEDED IN MoveL ACTION CALL.

class AttacherClient(Node):
    
    def __init__(self):
        # Initialise node:
        super().__init__('Attacher_client')
        self._action_client = ActionClient(self, Attacher, 'Attacher')
        # Wait for ATTACHER server to be available:
        print ("Waiting for ATTACHER action server to be available...")
        self._action_client.wait_for_server()
        print ("Attacher ACTION SERVER detected.")
    
    def send_goal(self, object, endeffector):
        # Assign variables:
        goal_msg = Attacher.Goal()
        goal_msg.object = object
        goal_msg.endeffector = endeffector
        # ACTION CALL:
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)

# DETACHER - Publihser:
class DetacherPub(Node):
    
    def __init__(self):
        # Declare NODE:
        super().__init__("ros2_PUBLISHER")
        # Declare PUBLISHER:
        self.publisher_ = self.create_publisher(String, "ros2_Detach", 5) #(msgType, TopicName, QueueSize)

# def getWaypoints(pose_start,pos_obj,pos_place):


# def getSequence(waypoints):


def main(args=None):
    # init ros node
    rclpy.init(args=args)
    
    # init clients
    MoveG_client = MoveGclient()
    attach_client = AttacherClient()
    detacher_client = DetacherPub()
    MoveXYZW_client = MoveXYZWclient()
    MoveL_client = MoveLclient()
    print("All clients initilaized...")

    time.sleep(1)

    # define poses 
    speed = 0.5
    # arm ready end-eff pose
    pose_ready = {'positionx': 0.70, 'positiony': 0.00, 'positionz': 1.50,
                    'yaw': -45.0, 'pitch': 0.0, 'roll': 180.0, 'speed': speed}
    
    # object pos
    pos_obj = {'positionx': 0.70, 'positiony': 0.30, 'positionz': 1.25, 'speed': speed}
    
    # place pos 
    pos_place = {'positionx': 0.70, 'positiony': -0.30, 'positionz': 1.25, 'speed': speed}
    
    # waypoints for pick and place
    # wp_pick0 = pos_obj.copy()
    # wp_pick0['positionz'] = 1.40
    wp_pick0 = {'movex': 0.00, 'movey': 0.30, 'movez': -0.10, 'speed': speed}
    
    # wp_pick = pos_obj.copy()
    # wp_pick['positionz'] = 1.235
    wp_pick = {'movex': 0.00, 'movey': 0.00, 'movez': -0.165, 'speed': speed}
    
    wp_pick1 = wp_pick.copy()
    wp_pick1['movez'] = -1*wp_pick1['movez']
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # wp_place0 = pos_place.copy()
    # wp_place0['positionz'] = 1.40
    wp_place0 = {'movex': 0.00, 'movey': -0.60, 'movez': -0.00, 'speed': speed}

    # wp_place = pos_place.copy()
    # wp_place['positionz'] = 1.235
    wp_place = {'movex': 0.00, 'movey': 0.00, 'movez': -0.165, 'speed': speed}

    wp_place1 = wp_place.copy()
    wp_place1['movez'] = -1*wp_place1['movez']

    # gripper open-close
    g_open = {'goal': 0.035}
    g_close = {'goal': 0.00}
    
    pose_spawn = {'positionx': 0.00, 'positiony': 0.00, 'positionz': 1.12,
                    'yaw': -90.0, 'pitch': -45.0, 'roll': -90.0, 'speed': 1.0}
    
    # Define sequence of actions
    # 1 = MoveXYZW
    # 2 = MoveL
    # 3 = MoveG
    sequence = [
        (1, pose_ready),
        (2, wp_pick0),
        (3, g_open),
        (2, wp_pick),
        (3, g_close),
        (2, wp_pick1),
        (2, wp_place0),
        (2, wp_place),
        (3, g_open),
        (2, wp_place1),
        (3, g_close),
        (1, pose_ready)
    ]

    for action, msg in sequence:
        global RES
        if action == 1:
            # Send goal to MoveXYZW action
            MoveXYZW_client.send_goal(msg)
            # spin once to get action result
            while rclpy.ok():   
                rclpy.spin_once(MoveXYZW_client)
                if (RES != "null"):
                    break
            if (RES == "MoveXYZW:SUCCESS"):
                RES = "null"
                time.sleep(1.5)
            else:
                print("ERROR: Action failed")
                break

        elif action == 2:
            # Send goal to MoveXYZW action
            MoveL_client.send_goal(msg)
            # spin once to get action result
            while rclpy.ok():   
                rclpy.spin_once(MoveL_client)
                if (RES != "null"):
                    break
            if (RES == "MoveL:SUCCESS"):
                RES = "null"
                time.sleep(1.5)
            else:
                print("ERROR: Action failed")
                break
        
        elif action == 3:
            # send goal to MoveG action
            MoveG_client.send_goal(msg)
            # spin once to get action result
            while rclpy.ok():   
                rclpy.spin_once(MoveG_client)
                if (RES != "null"):
                    break
            if (RES == "MoveG:SUCCESS"):
                RES = "null"
                time.sleep(0.5)
            else:
                print("ERROR: Action failed")
                break
        
        # time.sleep(1.5)
        # key_press = input()
        # while not key_press == '':
        #     pass

    print("SEQUENCE EXECUTION FINISHED!")

if __name__ == '__main__':
    main()
