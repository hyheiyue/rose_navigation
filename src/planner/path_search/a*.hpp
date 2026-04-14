#pragma once
#include "map/rose_map.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include <memory>
namespace rose_nav::planner {
class AStar {
public:
    enum SearchState : int { SUCCESS = 0, NO_PATH = 1, TIMEOUT = 2 };
    AStar(map::RoseMap::Ptr rose_map, const ParamsNode& config);
    using Ptr = std::unique_ptr<AStar>;
    static Ptr create(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        return std::make_unique<AStar>(rose_map, config);
    }
    ~AStar();
    using Path = std::vector<Eigen::Vector2d>;
    SearchState search(const Eigen::Vector2d& _start_w, const Eigen::Vector2d& _goal_w, Path& path);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace rose_nav::planner