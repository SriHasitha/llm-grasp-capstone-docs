#include <memory>
#include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"

using std::placeholders::_1;
class ImageSubscriber : public rclcpp::Node
{
  public:
    ImageSubscriber()
    : Node("image_subscriber")
    {
      // subscription_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      // "/realsense/points", 10, std::bind(&ImageSubscriber::pointcloud_callback, this, _1));
      
      subscription_rgbimage_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/color/image_raw", 10, std::bind(&ImageSubscriber::rgbimage_callback, this, _1));  

      subscription_depthimage_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/depth/image_raw", 10, std::bind(&ImageSubscriber::depthimage_callback, this, _1));      
    }

  private:
    // void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    // {
    //   RCLCPP_INFO(this->get_logger(), "Received point cloud with %d points", msg->width * msg->height);
    // }

    void rgbimage_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "Received rgb image with %d width and %d height", msg->width, msg->height);
    }    
    
    void depthimage_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "Received depth image with %d width and %d height", msg->width, msg->height);
    }
    
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_pointcloud_;
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
