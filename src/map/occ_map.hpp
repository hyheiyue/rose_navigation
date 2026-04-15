#pragma once

#include "utils/lock_queue.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include "voxel_map.hpp"
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <rclcpp/utilities.hpp>
#include <string>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
namespace rose_nav::map {
class OccMap {
public:
    struct Frame {
        double time;
        std::vector<Eigen::Vector3f> pts;
        Eigen::Vector3f sensor_origin;
    };
    using Ptr = std::shared_ptr<OccMap>;

    struct Cell {
        float log_odds = 0.f;
        double last_update = -1;
        void reset() {
            log_odds = 0.f;
        }
    };

    explicit OccMap(const ParamsNode& config);
    ~OccMap();
    static Ptr create(const ParamsNode& config) {
        return std::make_shared<OccMap>(config);
    }

    void insert_point_cloud(Frame&& frame) noexcept;
    inline bool is_occupied(int idx) const noexcept;

    void update(double now) noexcept;
    Eigen::Vector3f center() const;
    std::vector<Eigen::Vector4f> get_occupied_points() const noexcept;

    void set_center(const Eigen::Vector3f& o) noexcept;

    inline int key_to_index(const VoxelKey<3>& k) const noexcept;
    inline VoxelKey<3> index_to_key(int idx) const noexcept;

    inline VoxelKey<3> world_to_key(const Eigen::Vector3f& p) const noexcept;
    inline Eigen::Vector3f key_to_world(const VoxelKey<3>& k) const noexcept;
    std::vector<int> get_occupied_idx() const noexcept;
    SlidingVoxelMap<3, Cell>::Ptr get_voxel_map() const noexcept;
    struct LogCtx {
        double free_cost = 0;
        double hit_cost = 0;
        double ray_cost = 0;
        double receive_cost = 0;
        int receive_count = 0;
        int hit_count = 0;
        int free_count = 0;
        void reset() {
            free_cost = 0;
            hit_cost = 0;
            ray_cost = 0;
            receive_cost = 0;
            receive_count = 0;
            hit_count = 0;
            free_count = 0;
        }
    };
    LogCtx& get_log_ctx();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rose_nav::map
