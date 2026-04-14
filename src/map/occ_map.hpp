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
    ~OccMap() {
        hit_queue_.stop();
        frame_queue_.stop();
        free_queue_.stop();
        ray_queue_.stop();

        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        if (hit_thread_.joinable()) {
            hit_thread_.join();
        }
        if (free_thread_.joinable()) {
            free_thread_.join();
        }
        if (ray_thread_.joinable()) {
            ray_thread_.join();
        }
    }
    static Ptr create(const ParamsNode& config) {
        return std::make_shared<OccMap>(config);
    }

    void insert_point_cloud(Frame&& frame) noexcept;
    void receive() noexcept;
    void hit() noexcept;
    void free() noexcept;
    void ray() noexcept;
    inline bool is_occupied(int idx) const noexcept {
        if (idx < 0)
            return params_.unknown_is_occupied;
        const Cell& c = voxel_map_->grid[idx];

        return c.log_odds > params_.occ_th;
    }

    void update(double now) noexcept;
    Eigen::Vector3f center() const {
        return voxel_map_->center;
    }
    std::vector<Eigen::Vector4f>
    get_occupied_points(float sample_resolution_m = -0.05) const noexcept;
    void reset_a_cell(int idx) noexcept {
        std::lock_guard<std::mutex> lock(*cell_mutexes[idx]);
        voxel_map_->grid[idx].reset();
    }
    void set_center(const Eigen::Vector3f& o) noexcept;

    struct RayResult {
        std::vector<int> idx, cnt;

        inline void reserve(size_t n) {
            idx.reserve(n);
            cnt.reserve(n);
        }
        inline void clear() {
            idx.clear();
            cnt.clear();
        }

        inline void push(int i, int c) {
            idx.push_back(i);
            cnt.push_back(c);
        }

        inline size_t size() const {
            return idx.size();
        }
    };

    template<typename Policy>
    inline void dda_raycast_kernel(
        const VoxelKey<3>& o,
        const Eigen::Vector3f& d,
        size_t max_range_vox,
        const Policy& policy,
        RayResult& out
    ) noexcept {
        using Vec3 = Eigen::Array3f;
        constexpr float eps = 1e-6f;
        const float inf = std::numeric_limits<float>::infinity();
        Vec3 pos = Vec3(o.x(), o.y(), o.z()) + 0.5f;
        Eigen::Vector3i c(o.x(), o.y(), o.z());
        Vec3 dir = d.array();
        Eigen::Vector3i step = (dir > 0).cast<int>() - (dir < 0).cast<int>();
        Vec3 tDelta = (dir.abs() > eps).select((1.f / dir).abs(), Vec3::Constant(inf));
        Vec3 nextBoundary = (step.cast<float>().array() > 0)
                                .select(c.cast<float>().array() + 1.f, c.cast<float>().array());
        Vec3 tMax = (dir.abs() > eps).select((nextBoundary - pos) / dir, Vec3::Constant(inf));
        for (size_t i = 0; i < max_range_vox; ++i) {
            int axis;
            tMax.minCoeff(&axis);
            c[axis] += step[axis];
            tMax[axis] += tDelta[axis];
            int idx = key_to_index({ c.x(), c.y(), c.z() });
            if (idx < 0 || policy.should_stop(c.x(), c.y(), c.z()))
                break;
            policy.emit(idx, out);
        }
    }

    inline int key_to_index(const VoxelKey<3>& k) const noexcept {
        return voxel_map_->key_to_index(k);
    }

    inline VoxelKey<3> index_to_key(int idx) const noexcept {
        return voxel_map_->index_to_key(idx);
    }

    inline VoxelKey<3> world_to_key(const Eigen::Vector3f& p) const noexcept {
        return voxel_map_->world_to_key(p);
    }
    inline Eigen::Vector3f key_to_world(const VoxelKey<3>& k) const noexcept {
        return voxel_map_->key_to_world(k);
    }
    struct Params {
        float log_hit;
        float log_free;
        float log_min;
        float log_max;
        float occ_th;

        int min_shift;
        double timeout;
        float max_ray_range;

        bool unknown_is_occupied;
        bool use_ray;
        void load(const ParamsNode& config) {
            log_hit = config.declare<float>("log_hit");
            log_free = config.declare<float>("log_free");
            log_min = config.declare<float>("log_min");
            log_max = config.declare<float>("log_max");
            occ_th = config.declare<float>("occ_th");
            min_shift = config.declare<int>("min_shift");
            timeout = config.declare<double>("timeout");
            max_ray_range = config.declare<float>("max_ray_range");
            unknown_is_occupied = config.declare<bool>("unknown_is_occupied");
            use_ray = config.declare<bool>("use_ray");
        }
    } params_;
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
    } log_ctx_;
    template<class Ctx>
    struct IdxSnap {
        Ctx ctx;
        std::vector<int> indices;
        std::vector<uint16_t> count;
    };
    struct IdxBuf {
        std::vector<uint16_t> count;
        std::vector<uint32_t> stamp;
        std::vector<int> indices;

        IdxBuf() {}
        IdxBuf(size_t n) {
            resize(n);
        }
        inline void try_push(int idx, uint32_t now) {
            if (stamp[idx] != now) {
                stamp[idx] = now;
                indices.push_back(idx);
            }
            count[idx]++;
        }
        inline void try_push(int idx, int count_val, uint32_t now) {
            if (stamp[idx] != now) {
                stamp[idx] = now;
                indices.push_back(idx);
            }
            count[idx] += count_val;
        }
        inline void resize(size_t n) {
            stamp.resize(n, -1);
            count.resize(n, 0);
        }
        void reset() {
            std::fill(stamp.begin(), stamp.end(), -1);
            std::fill(count.begin(), count.end(), 0);
            indices.clear();
        }
        template<class Ctx>
        IdxSnap<Ctx> make_snap(Ctx ctx) {
            IdxSnap<Ctx> snap;
            snap.ctx = ctx;
            snap.count.reserve(indices.size());
            for (int idx: indices) {
                snap.count.push_back(count[idx]);
            }
            snap.indices = indices;

            return snap;
        }
    };
    SlidingVoxelMap<3, Cell>::Ptr voxel_map_;
    std::vector<std::unique_ptr<std::mutex>> cell_mutexes;
    std::vector<int> occupied_buffer_idx_;
    double now_;
    LockQueue<IdxSnap<double>> hit_queue_, free_queue_;
    struct RayCtx {
        double time;
        VoxelKey<3> sensor_key;
    };
    struct Ray {
        IdxSnap<RayCtx> inmap;
        std::vector<VoxelKey<3>> outmap;
    };
    LockQueue<Ray> ray_queue_;
    LockQueue<Frame> frame_queue_ { 10 };
    std::thread receive_thread_, hit_thread_, free_thread_, ray_thread_;
};

} // namespace rose_nav::map
