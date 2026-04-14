#pragma once
#include <rclcpp/node.hpp>
namespace rose_nav::lm {
class SmallPointLIO {
public:
    SmallPointLIO(rclcpp::Node& node);
    ~SmallPointLIO();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace rose_nav::lm