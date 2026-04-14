#pragma once

#include <rclcpp/node.hpp>
namespace rose_nav::planner {
class RosePlanner {
public:
    RosePlanner(rclcpp::Node& node);
    ~RosePlanner();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace rose_nav::planner