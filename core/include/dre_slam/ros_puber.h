#ifndef ROS_PUBER_H
#define ROS_PUBER_H

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap/octomap.h>
#include <octomap_ros/conversions.hpp>
#include <dre_slam/map.h>
#include <geometry_msgs/msg/pose_stamped.hpp>   // For geometry_msgs::msg::PoseStamped
#include <geometry_msgs/msg/pose_array.hpp>     // For geometry_msgs::msg::PoseArray
#include <sensor_msgs/msg/point_cloud2.hpp>     // For sensor_msgs::msg::PointCloud2
#include <visualization_msgs/msg/marker.hpp>    // For visualization_msgs::msg::Marker


namespace dre_slam {

class RosPuber {
public:
    // Change constructor to use rclcpp::Node::SharedPtr
    RosPuber(rclcpp::Node::SharedPtr node);

    void pubCurrentFrame(Frame* frame);
    void pubDynamicPixelCullingResults(KeyFrame* kf);
    void pubSparseMap(Map* map);
    void pubOctoMap(octomap::OcTree* octree);

private:
    rclcpp::Node::SharedPtr node_; // Store the node pointer

    // Info of the current frame.
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr puber_robot_pose_;
    image_transport::Publisher puber_img_match_;

    // Dynamic pixel detection results.
    image_transport::Publisher puber_dpc_img_objects_;
    image_transport::Publisher puber_dpc_img_clusters_;
    image_transport::Publisher puber_dpc_img_mask_;

    // KFs and pose graph. Sparse Map
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr puber_mappoints_;    
	rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr puber_kfs_puber_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_encoder_graph_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_loop_graph_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_visual_graph_;

    // OctoMap.
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr puber_octomap_;
}; // class RosPuber

} // namespace dre_slam

#endif // ROS_PUBER_H
