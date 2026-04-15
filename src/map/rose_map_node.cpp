#include "backward-cpp/backward.hpp"
#include "rose_map.hpp"
#include <rclcpp/rclcpp.hpp>
namespace rose_nav {
class RoseMapNode: public rclcpp::Node {
public:
    RoseMapNode(const rclcpp::NodeOptions& options): Node("rose_nav_map", options) {
        rose_map = std::make_unique<map::RoseMap>(*this);
    }
    std::unique_ptr<map::RoseMap> rose_map;
    backward::SignalHandling sh;
};
} // namespace rose_nav
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_nav::RoseMapNode)