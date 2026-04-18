#pragma once
#include "../common.hpp"
#include "utils/rclcpp_parameter_node.hpp"

#include "map/rose_map.hpp"
#include <memory>

namespace rose_nav::planner {

class TrajOpt {
public:
    using Ptr = std::unique_ptr<TrajOpt>;
    TrajOpt(map::RoseMap::Ptr rose_map, const ParamsNode& config);
    static Ptr create(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        return std::make_unique<TrajOpt>(rose_map, config);
    }
    ~TrajOpt();
    std::optional<TrajType> optimize(
        const std::vector<Eigen::Vector2d>& path,
        const RoboState& now,
        bool use_opt,
        std::optional<std::pair<int, int>> some_no_opt = std::nullopt
    );
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rose_nav::planner