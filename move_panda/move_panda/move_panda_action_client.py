import time
import rclpy
import sys
# from clients import *

from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from ros2_data.action import MoveXYZW, MoveG, MoveL, MoveXYZ
from ros2_grasping.action import Attacher
from langsam_interface.srv import BoundingBoxPrediction
from ggcnn_interface.srv import GraspPrediction

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

# LangSAM service client
BOUNDING_BOX = None
P_CHECK_LANGSAM = False
class LangsamClient(Node):
    def __init__(self):
        super().__init__('langsam_client')
        self.client = self.create_client(BoundingBoxPrediction, 'get_bounding_box')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('LangSAM not available, waiting again...')
        self.req = BoundingBoxPrediction.Request()

    def call_service(self, prompt):
        self.req.prompt = prompt
        self.get_logger().info('Calling LangSAM service ...')
        future = self.client.call_async(self.req)
        future.add_done_callback(self.callback)

    def callback(self, future):
        global BOUNDING_BOX
        global P_CHECK_LANGSAM
        try:
            response = future.result()
            self.get_logger().info('Received bounding boxes')
            BOUNDING_BOX = response
            P_CHECK_LANGSAM = True
            # return response.boxes
        except Exception as e:
            self.get_logger().info('LangSAM service call failed %r' % (e,))
            # return None

POSE = None
P_CHECK_POSE = False
# GGCNN service client
class GGCNNclient(Node):
    def __init__(self):
        super().__init__('ggcnn_client')
        self.client = self.create_client(GraspPrediction,'grasp_prediction')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('GGCNN not available, waiting again...')
        self.req = GraspPrediction.Request()

    def call_service(self,bbox):
        self.req.box.xmin = bbox.xmin
        self.req.box.ymin = bbox.ymin
        self.req.box.xmax = bbox.xmax
        self.req.box.ymax = bbox.ymax
        future = self.client.call_async(self.req)
        self.get_logger().info('Calling GGCNN service ...')
        future.add_done_callback(self.callback)

    def callback(self,future):
        global POSE
        global P_CHECK_POSE
        try:
            response = future.result()
            if response.success == True:
                self.get_logger().info('Received predicted grasp pose')
                POSE = response.best_grasp
                P_CHECK_POSE = True
            else:
                self.get_logger().info('GGCNN failed to predict grasp pose')
                POSE = None
                P_CHECK_POSE = True
        except Exception as e:
            self.get_logger().info('GGCNN service call failed %r' % (e,))
        
# Input prompt as a ROS2 parameter
PROMPT = "default"
P_CHECK_PROMPT = False
class promptPARAM(Node):
    def __init__(self):
        global PROMPT
        global P_CHECK_PROMPT
        super().__init__('param_node')

        self.declare_parameter('prompt', "default")
        PROMPT = self.get_parameter('prompt').get_parameter_value().string_value
        if (PROMPT == "default"):
            self.get_logger().info('Prompt was not given.')
            exit()
        else:    
            self.get_logger().info('Prompt received: ' + PROMPT)
        P_CHECK_PROMPT = True 

def main(args=None):
    # init ros node
    rclpy.init(args=args)
    
    # extract prompt from ROS2 params
    global PROMPT
    global P_CHECK_PROMPT

    paramNode = promptPARAM()
    while (P_CHECK_PROMPT==False):
        rclpy.spin_once(paramNode)
    paramNode.destroy_node()

    # init clients
    MoveG_client = MoveGclient()
    attach_client = AttacherClient()
    detacher_client = DetacherPub()
    MoveXYZW_client = MoveXYZWclient()
    MoveL_client = MoveLclient()
    ggcnn_client = GGCNNclient()
    langsam_client = LangsamClient()
    print("All clients initilaized...")

    time.sleep(1)

    ################ Call LangSAM ################
    global BOUNDING_BOX
    global P_CHECK_LANGSAM
    langsam_client.call_service(PROMPT)
    while (P_CHECK_LANGSAM == False):
        rclpy.spin_once(langsam_client)
    if (BOUNDING_BOX == None):
        print('Exiting')
        exit()
    else:
        print(BOUNDING_BOX)       
    langsam_client.destroy_node()

    time.sleep(1)

    ################ Call GGCNN #################
    global POSE
    global P_CHECK_POSE
    ggcnn_client.call_service(BOUNDING_BOX.boxes[0])
    while(P_CHECK_POSE == False):
        rclpy.spin_once(ggcnn_client)
    if (POSE == None):
        print('Exiting')
        exit()
    else:
        print(POSE)
    ggcnn_client.destroy_node()

    time.sleep(1)

    ############### Pick & Place ################ 
    # define poses 
    speed = 0.5
    # arm ready end-eff pose
    pose_ready = {'positionx': POSE.pose.position.x, 'positiony': (POSE.pose.position.y), 'positionz': POSE.pose.position.z + 0.35,
                    'yaw': -45.0, 'pitch': 0.0, 'roll': 180.0, 'speed': speed} #roll: 180.0, pitch: 0.0, yaw: -45 x: 0.70, y: 0.00, z: 1.50
    
    # object pos
    pos_obj = {'positionx': 0.70, 'positiony': 0.30, 'positionz': 1.25, 'speed': speed}
    
    # place pos 
    pos_place = {'positionx': 0.70, 'positiony': -0.30, 'positionz': 1.25, 'speed': speed}
    
    # waypoints for pick and place
    # wp_pick0 = pos_obj.copy()
    # wp_pick0['positionz'] = 1.40
    wp_pick0 = {'movex': 0.00, 'movey': 0.00, 'movez': -0.10, 'speed': speed}
    
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
