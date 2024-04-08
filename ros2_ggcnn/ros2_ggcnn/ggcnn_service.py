import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from ggcnn_interface.srv import GraspPrediction
from ros2_ggcnn.ggcnn_torch import process_depth_image,predict

import numpy as np
import cv2
from cv_bridge import CvBridge

cv_bridge = CvBridge()

class GGCNNservice(Node):
    def __init__(self):
        super().__init__('ggcnn_service')

        # init image subscribers
        self.rgb_sub = self.create_subscription(Image,'/camera/image_raw',self.rgb_callback,1)
        self.depth_sub = self.create_subscription(Image,'/camera/depth_raw',self.depth_callback,1)
        self.get_logger().info('Initialized subscribers')
        # init ggcnn service
        self.srv = self.create_service(GraspPrediction,'grasp_prediction',self.grasp_prediction_callback)
        self.get_logger().info('Initialized service')

        # to check if depth image was received
        self.received = False
        # self.depth = np.zeros((300,300))

    def rgb_callback(self,img_msg):
        self.rbg = cv_bridge.imgmsg_to_cv2(img_msg)
        self.get_logger().info('Recieved RGB image: ',img_msg.height,'X',img_msg.width)

    def depth_callback(self,img_msg):
        self.depth = cv_bridge.imgmsg_to_cv2(img_msg)
        self.get_logger().info('Received depth image: ',img_msg.height,'X',img_msg.width)
        self.received = True

    def grasp_prediction_callback(self,request,response):
        self.get_logger().info('Predicting Grasp Pose')
        depth = self.depth.copy()
        # crop depth image
        depth_crop = process_depth_image(depth)
        # predict grasp
        points,angle,width_img,_ = predict(depth_crop)

        # convert to 3d pose
        best_grasp_pose = []
        
        # construct response
        response.success = True
        # g = response.best_grasp
        # g.pose.position.x = 0.0
        # g.pose.position.y = 0.0
        # g.pose.position.z = 0.0
        # g.pose.orientation = 0.0
        # g.width = 0.0
        # g.quality = 0.0

        return response

def main():
    rclpy.init()
    ggcnn_service = GGCNNservice()
    rclpy.spin(ggcnn_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()