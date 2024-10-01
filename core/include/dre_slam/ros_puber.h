#ifndef ROS_PUBER_H
#define ROS_PUBER_H

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <octomap_msgs/Octomap.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.hpp>
#include <dre_slam/map.h>

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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr puber_mappoints_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr puber_kfs_puber_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_encoder_graph_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_loop_graph_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr puber_visual_graph_;

    // OctoMap.
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr puber_octomap_;
}; // class RosPuber

} // namespace dre_slam

#endif // ROS_PUBER_H
