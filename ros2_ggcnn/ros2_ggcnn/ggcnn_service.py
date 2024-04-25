import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image,CameraInfo
from ggcnn_interface.srv import GraspPrediction
from ros2_ggcnn.ggcnn_torch import process_depth_image,predict

import numpy as np
import cv2
from cv_bridge import CvBridge
import time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped,Quaternion
from scipy.spatial.transform import Rotation as R

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

        # init transform listner
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # depth camera params
        self.K_depth = []
        self.depth_scale = 1000
        self.h_depth = 0
        self.w_depth = 0
        # to check if msgs are received
        self.received_depth = False
        self.received_rgb = False
        self.received_depth_K = False
        self.received_transformation = False
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

    # function to obtain transformation between frames       
    def tf_transformation(self,from_frame,to_frame):
        self.time = rclpy.time.Time()
        count = 0
        while ( (not self.received_transformation) and (count < 10) ):
            try:
                transformation: TransformStamped = self.tf_buffer.lookup_transform(to_frame, from_frame, self.time)
                self.get_logger().info(f'Received Transform: {transformation}')
                self.received_transformation = True
            except TransformException:
                self.get_logger().info('Error: Unable to receive trasnform')
                time.sleep(1)
                continue
            count += 1
        return transformation

    def show_image(self,img,title):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')
        cv2.imshow(title, img)
        cv2.waitKey() 
    
    def grasp_prediction_callback(self,request,response):
        # Wait for the depth image to be received
        while not self.received_depth:
            self.get_logger().info('Waiting for depth image...')
            rclpy.spin_once(self, timeout_sec=1)  # Briefly yield control to allow other callbacks to run

        # crop params
        crop_size = 300
        out_size = 300
        crop_offset = 0
        # crop depth image
        depth = self.depth.copy()
        depth_crop = process_depth_image(depth,crop_size=crop_size,out_size=out_size,crop_y_offset=crop_offset)
        # show images
        self.show_image(depth,'Depth Image')
        # self.get_logger().info('waiting to print cropped depth image')
        # self.get_logger().info(f'cropped depth image: {self.depth_crop}')
        # self.get_logger().info('waiting to print cropped depth image')
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
        x_cam = (max_pixel[1] - cx)/(fx) * point_depth  # meters
        y_cam = (max_pixel[0] - cy)/(fy) * point_depth  # meters
        z_cam = -point_depth                            # meters

        # Transformation mat of obj wrt camera frame
        T_grasp_cam = np.array([[np.cos(ang), -np.sin(ang), 0 , x_cam],
                              [np.sin(ang),  np.cos(ang), 0 , y_cam],
                              [0          ,  0          , 1 , z_cam],
                              [0          ,  0          , 0 , 1    ]])

        camera_pose = self.tf_transformation('d435_depth_frame','base_link')
        # camera position wrt frame frame
        P_cam_world = np.array([camera_pose.transform.translation.x, camera_pose.transform.translation.y, camera_pose.transform.translation.z])
        # camera rot mat wrt world frame
        R_cam_world = R.as_matrix(R.from_quat([ camera_pose.transform.rotation.x,
                                                camera_pose.transform.rotation.y,
                                                camera_pose.transform.rotation.z,
                                                camera_pose.transform.rotation.w]))
        # transformation mat of cam wrt world frame
        T_cam_world = np.hstack((R_cam_world,P_cam_world.reshape(-1,1)))
        T_cam_world = np.vstack((T_cam_world,np.array([0,0,0,1])))

        # transform grasp pose to world frame
        T_grasp_world = T_grasp_cam @ T_cam_world
        # extract pose
        pos = T_grasp_world[:3,3]
        self.get_logger().info(f'xyz in world frame: {pos}')
        # calc quaternion from rot mat
        rot = R.from_matrix(T_grasp_world[:3,:3])
        matrix = rot.as_matrix()
        # Calculate roll (x-axis rotation)
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        # Calculate pitch (y-axis rotation)
        pitch = -np.arcsin(matrix[2, 0])
        # Calculate yaw (z-axis rotation)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        self.get_logger().info(f'roll: {roll},pitch: {pitch}, yaw: {yaw}')
        quat = rot.as_quat()
        self.get_logger().info(f'quaternion; {quat}')

        # construct response
        response.success = True
        quat_msg = Quaternion()
        quat_msg.x = quat[0]
        quat_msg.y = quat[1]
        quat_msg.z = quat[2]
        quat_msg.w = quat[3]
        g = response.best_grasp
        g.pose.position.x = pos[0]
        g.pose.position.y = pos[1]
        g.pose.position.z = pos[2]
        g.pose.orientation = quat_msg
        g.width = width.astype(float)
        # g.quality = 0.0

        return response

def main():
    rclpy.init()
    ggcnn_service = GGCNNservice()
    rclpy.spin(ggcnn_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
