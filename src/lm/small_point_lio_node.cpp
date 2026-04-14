#include "small_point_lio.hpp"
#include <rclcpp/rclcpp.hpp>
namespace rose_nav {
class SmallPointLioNode: public rclcpp::Node {
public:
    SmallPointLioNode(const rclcpp::NodeOptions& options): Node("rose_nav_lm", options) {
        small_point_lio = std::make_unique<lm::SmallPointLIO>(*this);
    }
    std::unique_ptr<lm::SmallPointLIO> small_point_lio;
};
} // namespace rose_nav
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_nav::SmallPointLioNode)