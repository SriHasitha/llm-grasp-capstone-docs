import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image,CameraInfo
from ggcnn_interface.srv import GraspPrediction
from ros2_ggcnn.ggcnn_torch import process_depth_image,predict

import numpy as np
import cv2
from cv_bridge import CvBridge

class GGCNNservice(Node):
    def __init__(self):
        super().__init__('ggcnn_service')

        # init image subscribers
        self.rgb_sub = self.create_subscription(Image,'/color/image_raw',self.rgb_callback,1)
        self.depth_sub = self.create_subscription(Image,'/depth/image_raw',self.depth_callback,1)
        # init cam_info subscriber to obtain intrinsic matrix
        self.depth_info_sub = self.create_subscription(CameraInfo,'/depth/camera_info',self.depth_info_callback,1)
        self.get_logger().info('Initialized subscribers')
        
        # init ggcnn service
        self.srv = self.create_service(GraspPrediction,'grasp_prediction',self.grasp_prediction_callback)
        self.get_logger().info('Initialized service')

        # depth camera params
        self.K_depth = []
        self.depth_scale = 1000
        self.h_depth = 0
        self.w_depth = 0

        # to check if depth image was received
        self.received_depth = False
        self.received_rgb = False
        self.received_depth_K = False
        # cv bridge to convert imgmsg to cv2img
        self.cv_bridge = CvBridge()

    def rgb_callback(self,img_msg):
        self.rbg = self.cv_bridge.imgmsg_to_cv2(img_msg)
        if not self.received_rgb:
            self.get_logger().info(f'Received RGB image: {img_msg.width}X{img_msg.height}')
        self.received_rgb = True

    def depth_callback(self,img_msg):
        self.depth = self.cv_bridge.imgmsg_to_cv2(img_msg)
        self.h_depth = img_msg.height
        self.w_depth = img_msg.width

        if not self.received_depth:
            self.get_logger().info(f'Received depth image: {self.w_depth}X{self.h_depth}')
        self.received_depth = True
    
    def depth_info_callback(self,info_msg):
        self.K_depth = info_msg.k
        if not self.received_depth_K:
            self.get_logger().info(f'Received Depth Camera Intrinsic Matrix:\n {self.K_depth}')
        self.received_depth_K = True

    def show_image(self,img,title):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')
        cv2.imshow(title, img)
        cv2.waitKey() 
    
    def grasp_prediction_callback(self,request,response):
        # Wait for the depth image to be received
        while not self.received_depth:
            self.get_logger().info('Waiting for depth image...')
            rclpy.spin_once(self, timeout_sec=0.1)  # Briefly yield control to allow other callbacks to run

        # crop params
        crop_size = 300
        out_size = 300
        crop_offset = 0
        # crop depth image
        depth = self.depth.copy()
        depth_crop = process_depth_image(depth,crop_size=crop_size,out_size=out_size,crop_y_offset=crop_offset)
        # show images
        self.show_image(depth,'Depth Image')
        self.show_image(depth_crop,'Cropped Depth Image')

        # predict grasp
        self.get_logger().info('Predicting Grasp Pose')
        points_out,angle_out,width_out = predict(depth_crop)

        # Calculate  depth
        # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
        depth_center = depth_crop[100:141, 130:171].flatten()
        depth_center.sort()
        depth_center = depth_center[:10].mean() * self.depth_scale

        # Get max quality pixel index from points_out
        max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))

        # get angle and width corresponding to above max_pixel
        ang = angle_out[max_pixel[0], max_pixel[1]]
        width = width_out[max_pixel[0], max_pixel[1]]

        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        max_pixel = ((np.array(max_pixel) / out_size * crop_size) + np.array([(self.h_depth - crop_size)//2 - crop_offset, (self.w_depth - crop_size) // 2]))
        max_pixel = np.round(max_pixel).astype(int)
        self.get_logger().info(f'max_pixel: {max_pixel}')

        point_depth = depth[max_pixel[0], max_pixel[1]]/self.depth_scale  # depth value in meters

        # Compute the actual position.
        fx = self.K_depth[0]
        fy = self.K_depth[4]
        cx = self.K_depth[2]
        cy = self.K_depth[5]
        x = (max_pixel[1] - cx)/(fx) * point_depth  # meters
        y = (max_pixel[0] - cy)/(fy) * point_depth  # meters
        z = point_depth                             # meters


        # convert to 3d pose
        best_grasp_pose = []
        
        # construct response
        response.success = True
        g = response.best_grasp
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = z.astype(float)
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