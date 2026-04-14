#pragma once
#include "map/rose_map.hpp"
#include "planner/common.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include <memory>
namespace rose_nav::planner {
class LMPC {
public:
    using Ptr = std::unique_ptr<LMPC>;
    LMPC(map::RoseMap::Ptr rose_map, const ParamsNode& config);
    static Ptr create(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        return std::make_unique<LMPC>(rose_map, config);
    }
    std::optional<ControlOutput> solve(std::chrono::duration<double> dt);
    void set_traj(const TrajType& traj);
    void set_current(const RoboState& c);
    ~LMPC();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace rose_nav::planner