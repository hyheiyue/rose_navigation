#pragma once
#include <Eigen/Dense>
#include <rclcpp/node.hpp>
namespace rose_nav {
class ParamsNode {
public:
    ParamsNode(rclcpp::Node& node, std::string prefix = ""): node_(node) {
        if (!prefix.empty() && prefix.back() != '.')
            prefix += ".";
        prefix_ = std::move(prefix);
    }

    template<typename T>
    [[nodiscard]] T declare(const std::string& name, const T& default_value) const {
        return node_.declare_parameter(prefix_ + name, default_value);
    }
    template<typename T>
    [[nodiscard]] T declare(const std::string& name) const {
        return node_.declare_parameter<T>(prefix_ + name);
    }

    [[nodiscard]] ParamsNode sub(const std::string& sub_prefix) const {
        return ParamsNode(node_, prefix_ + sub_prefix);
    }

private:
    rclcpp::Node& node_;
    std::string prefix_;
};
} // namespace rose_nav
