// Include necessary ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <dre_slam/dre_slam.h>
#include <sys/stat.h>

using namespace dre_slam;

class SensorGrabber : public rclcpp::Node
{
public:
	// Constructor to initialize the SLAM pointer and create ROS2 subscriptions
	SensorGrabber()
		: Node("dre_slam_node"), 
		last_enl_(0), 
		last_enr_(0),
		dre_slam_cfg_dir_("/root/catkin_ws/src/dre_slam/config/comparitive_test.yaml"),
		orbvoc_dir_("/root/catkin_ws/src/dre_slam/config/orbvoc/ORBvoc.bin"),
        yolov3_classes_dir_("/root/catkin_ws/src/dre_slam/config/yolov3/coco.names"),
        yolov3_model_dir_("/root/catkin_ws/src/dre_slam/config/yolov3/yolov3.cfg"),
        yolov3_weights_dir_("/root/catkin_ws/src/dre_slam/config/yolov3/yolov3.weights"),
        results_dir_("/root/results")
	{

		// Create subscribers with message filters
		rgb_sub_.subscribe(this, "/camera/rgb/image_raw", rmw_qos_profile_sensor_data);
		depth_sub_.subscribe(this, "/camera/depth/image_raw", rmw_qos_profile_sensor_data);

		// Set up synchronization policy (ApproximateTime)
		sync_ = std::make_shared<message_filters::Synchronizer<sync_policy_t>>(sync_policy_t(10), rgb_sub_, depth_sub_);
		sync_->registerCallback(&SensorGrabber::grabRGBD, this);

		// Subscription for Encoder
		encoder_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
			"/encoder", 10, std::bind(&SensorGrabber::grabEncoder, this, std::placeholders::_1));

		// Initialize the SLAM system
		Config* cfg = new Config(dre_slam_cfg_dir_);
		slam_ = new DRE_SLAM(this, cfg, orbvoc_dir_, yolov3_classes_dir_, yolov3_model_dir_, yolov3_weights_dir_);
	}

	// Callback to handle RGBD data
	void grabRGBD(const sensor_msgs::msg::Image::SharedPtr msg_rgb,
				  const sensor_msgs::msg::Image::SharedPtr msg_depth)
	{
		if (!msg_rgb || !msg_depth)
			return;

		// Get images using cv_bridge
		auto cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb);
		auto cv_ptr_depth = cv_bridge::toCvShare(msg_depth);

		// Add RGB-D images to SLAM system
		slam_->addRGBDImage(cv_ptr_rgb->image, cv_ptr_depth->image, cv_ptr_rgb->header.stamp.sec);
	}

	// Callback to handle Encoder data
	void grabEncoder(const geometry_msgs::msg::QuaternionStamped::SharedPtr en_ptr)
	{
		double enl1 = en_ptr->quaternion.x;
		double enl2 = en_ptr->quaternion.y;
		double enr1 = en_ptr->quaternion.z;
		double enr2 = en_ptr->quaternion.w;

		double enl = 0.5 * (enl1 + enl2);
		double enr = 0.5 * (enr1 + enr2);
		double ts = en_ptr->header.stamp.sec;

		if (last_enl_ == 0 && last_enr_ == 0)
		{
			last_enl_ = enl;
			last_enr_ = enr;
			return;
		}

		double delta_enl = fabs(enl - last_enl_);
		double delta_enr = fabs(enr - last_enr_);

		const double delta_th = 4000;
		if (delta_enl > delta_th || delta_enr > delta_th)
		{
			RCLCPP_INFO(this->get_logger(), "Encoder jump detected");
			return;
		}

		last_enl_ = enl;
		last_enr_ = enr;

		slam_->addEncoder(enl, enr, ts);
	}

private:
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> sync_policy_t;
	std::shared_ptr<message_filters::Synchronizer<sync_policy_t>> sync_;

	DRE_SLAM *slam_;
	double last_enl_, last_enr_;

	std::string dre_slam_cfg_dir_;
	std::string orbvoc_dir_;
	std::string yolov3_classes_dir_;
	std::string yolov3_model_dir_;
	std::string yolov3_weights_dir_;
	std::string results_dir_;

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
	rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr encoder_sub_;
};

// Main function to initialize the node and start processing
int main(int argc, char **argv)
{
	// Initialize the ROS 2 node
	rclcpp::init(argc, argv);

	// Create an instance of the SLAM system and the SensorGrabber node
	auto node = std::make_shared<SensorGrabber>();

	// Start spinning and processing callbacks
	rclcpp::spin(node);

	rclcpp::shutdown();

	return 0;
}
