#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
// for connection between opencv-ros
#include <image_transport/image_transport.hpp>
#include "cv_bridge/cv_bridge.h"
// for opencv functions
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using std::placeholders::_1;
class ImageSubscriber : public rclcpp::Node
{
  public:
    ImageSubscriber()
    : Node("image_subscriber")
    {   
      subscription_rgbimage_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/color/image_raw", 10, std::bind(&ImageSubscriber::rgbimage_callback, this, _1));  

      subscription_depthimage_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/depth/image_raw", 10, std::bind(&ImageSubscriber::depthimage_callback, this, _1));      
    }

  private:
    void rgbimage_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "Received rgb image with %d width and %d height", msg->width, msg->height);
      // for converting ros data to opencv data
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

      std::string rgb_image_path = "/home/hasithab/llm-grasping-panda/src/llm-grasp-capstone-docs/gazebo_images/rgb/";
      cv::imwrite(rgb_image_path + "rgb_image_" + std::to_string(msg->header.stamp.sec) + ".png", cv_ptr->image);
      RCLCPP_INFO(this->get_logger(), "Saved RGB image to: %s", rgb_image_path.c_str());
    }    
    
    void depthimage_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "Received depth image with %d width and %d height", msg->width, msg->height);
      // for converting ros data into opencv data
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);

      // Normalize the depth image for proper visualization
      cv::Mat depth_image_normalized;
      cv_ptr->image.convertTo(depth_image_normalized, CV_8U, 255.0 / 1000.0); // Scale to 0-255 range (assuming max depth is 1000mm)


      std::string depth_image_path = "/home/hasithab/llm-grasping-panda/src/llm-grasp-capstone-docs/gazebo_images/depth/";
      cv::imwrite(depth_image_path + "depth_image_" + std::to_string(msg->header.stamp.sec) + ".png", depth_image_normalized); 
      RCLCPP_INFO(this->get_logger(), "Saved depth image to: %s", depth_image_path.c_str());
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_rgbimage_;	
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_depthimage_;	
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv); 
  rclcpp::spin(std::make_shared<ImageSubscriber>());
  rclcpp::shutdown();
  return 0;
}