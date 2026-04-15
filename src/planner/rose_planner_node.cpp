#include "backward-cpp/backward.hpp"
#include "rose_planner.hpp"
#include <rclcpp/rclcpp.hpp>
namespace rose_nav {
class RosePlannerNode: public rclcpp::Node {
public:
    RosePlannerNode(const rclcpp::NodeOptions& options): Node("rose_nav_planner", options) {
        rose_planner = std::make_unique<planner::RosePlanner>(*this);
    }
    std::unique_ptr<planner::RosePlanner> rose_planner;
    backward::SignalHandling sh;
};
} // namespace rose_nav
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_nav::RosePlannerNode)