#include "occ_map.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <thread>
#include <utility>
#include <vector>
namespace rose_nav::map {
struct OccMap::Impl {
    Impl(const ParamsNode& config) {
        params_.load(config);
        auto voxel_map_config = config.sub("voxel_map");
        float voxel_size = voxel_map_config.declare<double>("voxel_size");
        auto size_vec = voxel_map_config.declare<std::vector<double>>("size");
        auto size = Eigen::Vector3f(size_vec[0], size_vec[1], size_vec[2]);
        auto center_vec = voxel_map_config.declare<std::vector<double>>("center");
        auto center = Eigen::Vector3f(center_vec[0], center_vec[1], center_vec[2]);
        voxel_map_ = SlidingVoxelMap<3, Cell>::create(voxel_size, size, center);
        const auto N = voxel_map_->grid_size();
        cell_mutexes_.resize(N);
        for (auto& m: cell_mutexes_) {
            m = std::make_unique<std::mutex>();
        }
        occupied_buffer_idx_.reserve(N);
        occupied_pos_.assign(N, -1);
        receive_thread_ = std::thread(std::bind(&OccMap::Impl::receive, this));
        free_thread_ = std::thread(std::bind(&OccMap::Impl::free, this));
        if (params_.use_ray) {
            ray_thread_ = std::thread(std::bind(&OccMap::Impl::ray, this));
        }
    }
    ~Impl() {
        frame_queue_.stop();
        free_queue_.stop();
        ray_queue_.stop();

        if (receive_thread_.joinable()) {
            receive_thread_.join();
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

    void insert_point_cloud(Frame&& frame) noexcept {
        frame_queue_.push(std::move(frame));
    }
    void receive() noexcept {
        const auto N = voxel_map_->grid_size();

        std::unique_ptr<IdxBuf> ray_buf;
        if (params_.use_ray) {
            ray_buf = std::make_unique<IdxBuf>(N);
        }
        uint32_t stamp = 0;
        while (rclcpp::ok()) {
            Frame frame;
            frame_queue_.wait_and_pop(frame);
            auto start = std::chrono::steady_clock::now();
            stamp++;
            auto sensor_center = frame.sensor_origin;
            auto& pts = frame.pts;

            Ray ray;
            if (ray_buf) {
                ray_buf->_reset();
                ray_buf->indices.reserve(pts.size());
                ray.outmap.reserve(pts.size());
            }
            for (const auto& pt: pts) {
                const VoxelKey<3> k_hit = world_to_key(pt);

                int idx = key_to_index(k_hit);
                if (idx >= 0) {
                    auto cell = get_cell(idx);
                    auto& c = cell.get();
                    c.log_odds = std::max(c.log_odds + params_.log_hit, params_.log_min);
                    c.last_update = frame.time;
                    track_state(idx);
                }
                if (ray_buf) {
                    if (idx >= 0) {
                        ray_buf->try_push(idx, stamp);
                    } else {
                        ray.outmap.push_back(k_hit);
                    }
                }
            }

            if (ray_buf) {
                ray.inmap = ray_buf->make_snap(RayCtx {
                    .time = frame.time,
                    .sensor_key = world_to_key(sensor_center),

                });
                ray_queue_.push(std::move(ray));
            }
            auto end = std::chrono::steady_clock::now();
            log_ctx_.receive_cost += std::chrono::duration<double, std::milli>(end - start).count();
            log_ctx_.receive_count++;
        }
    }

    void free() noexcept {
        while (rclcpp::ok()) {
            IdxSnap<double> free;
            free_queue_.wait_and_pop(free);
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < free.indices.size(); ++i) {
                int idx = free.indices[i];

                auto cell = get_cell(idx);
                auto& c = cell.get();

                c.log_odds =
                    std::min(c.log_odds + params_.log_free * free.count[i], params_.log_max);
                c.last_update = free.ctx;
                track_state(idx);
            }
            // tbb::parallel_for(
            //     size_t(0),
            //     free.indices.size(),
            //     [&](size_t i) {
            //         int idx = free.indices[i];

            //         auto cell = get_cell(idx);
            //         auto& c = cell.get();

            //         c.log_odds =
            //             std::min(c.log_odds + params_.log_free * free.count[i], params_.log_max);
            //         c.last_update = free.ctx;
            //         track_state(idx);
            //     },
            //     tbb::auto_partitioner()
            // );
            auto end = std::chrono::steady_clock::now();
            log_ctx_.free_cost += std::chrono::duration<double, std::milli>(end - start).count();
            log_ctx_.free_count += free.indices.size();
        }
    }
    void ray() noexcept {
        const auto N = voxel_map_->grid_size();
        const size_t max_range_vox =
            static_cast<size_t>(std::ceil(params_.max_ray_range / voxel_map_->voxel_size));

        uint32_t stamp = 0;
        IdxBuf free(N);

        while (rclcpp::ok()) {
            Ray ray;
            ray_queue_.wait_and_pop(ray);

            auto start = std::chrono::steady_clock::now();
            stamp++;

            const auto o = ray.inmap.ctx.sensor_key;
            const Eigen::Vector3f o_center(o.x() + 0.5f, o.y() + 0.5f, o.z() + 0.5f);

            tbb::enumerable_thread_specific<RayResult> tls([&] {
                RayResult v;
                v.reserve(256);
                return v;
            });
            struct DDARayPolicy {
                int tx, ty, tz, count;

                inline bool should_stop(int x, int y, int z) const {
                    return x == tx && y == ty && z == tz;
                }

                inline void emit(int idx, RayResult& out) const {
                    out.push(idx, count);
                }
            };

            tbb::parallel_for(
                size_t(0),
                ray.inmap.indices.size(),
                [&](size_t i) {
                    auto& out = tls.local();

                    const int idx = ray.inmap.indices[i];
                    const auto k = index_to_key(idx);

                    Eigen::Vector3f d(k.x() + 0.5f, k.y() + 0.5f, k.z() + 0.5f);
                    d -= o_center;

                    const int cnt = ray.inmap.count[i];

                    DDARayPolicy policy {
                        .tx = k.x(),
                        .ty = k.y(),
                        .tz = k.z(),
                        .count = cnt,
                    };

                    dda_raycast_kernel(o, d, max_range_vox, policy, out);
                },
                tbb::auto_partitioner()
            );
            struct DDAOutmapPolicy {
                inline bool should_stop(int, int, int) const {
                    return false;
                }
                inline void emit(int idx, RayResult& out) const {
                    out.push(idx, 1);
                }
            };
            tbb::parallel_for(
                size_t(0),
                ray.outmap.size(),
                [&](size_t i) {
                    auto& out = tls.local();

                    const auto& h = ray.outmap[i];

                    Eigen::Vector3f d(h.x() + 0.5f, h.y() + 0.5f, h.z() + 0.5f);
                    d -= o_center;

                    DDAOutmapPolicy policy;
                    dda_raycast_kernel(o, d, max_range_vox, policy, out);
                },
                tbb::auto_partitioner()
            );

            free._reset();
            free.indices.reserve(N);

            for (auto& local: tls) {
                for (size_t i = 0; i < local.size(); ++i) {
                    free.try_push(local.idx[i], local.cnt[i], stamp);
                }
            }

            free_queue_.push(std::move(free.make_snap(ray.inmap.ctx.time)));

            auto end = std::chrono::steady_clock::now();
            log_ctx_.ray_cost +=
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
    }
    inline bool is_occupied(int idx) const noexcept {
        if (idx < 0)
            return params_.unknown_is_occupied;
        const Cell& c = voxel_map_->grid[idx];

        return c.log_odds > params_.occ_th && std::abs(c.last_update - now_) < params_.timeout;
    }

    void update(double now) noexcept {
        now_ = now;
        auto& grid = voxel_map_->grid;
        const auto N = grid.size();

        auto occ_buf_copy = get_occupied_idx();
        for (const auto& idx: occ_buf_copy) {
            track_state(idx);
        }
    }
    Eigen::Vector3f center() const {
        return voxel_map_->center;
    }
    std::vector<Eigen::Vector4f> get_occupied_points() const noexcept {
        std::vector<Eigen::Vector4f> pts;
        auto occ_buf = get_occupied_idx();
        pts.reserve(occ_buf.size());

        for (size_t i = 0; i < occ_buf.size(); i++) {
            int idx = occ_buf[i];
            auto p = key_to_world(index_to_key(idx));
            pts.emplace_back(p.x(), p.y(), p.z(), p.z() - voxel_map_->center.z());
        }

        return pts;
    }

    void set_center(const Eigen::Vector3f& o) noexcept {
        VoxelKey<3> new_center = world_to_key(o);
        VoxelKey<3> shift { new_center.x() - voxel_map_->center_key.x(),
                            new_center.y() - voxel_map_->center_key.y(),
                            new_center.z() - voxel_map_->center_key.z() };

        const int min_shift = params_.min_shift;
        if (std::abs(shift.x()) < min_shift && std::abs(shift.y()) < min_shift
            && std::abs(shift.z()) < min_shift)
            return;

        voxel_map_->slide_to(new_center, [&](int idx) { reset_a_cell(idx); });
    }
    class CellGuard {
    public:
        CellGuard(Cell& cell, std::mutex& mtx): lock_(mtx), cell_(cell) {}

        Cell& get() {
            return cell_;
        }
        Cell* operator->() {
            return &cell_;
        }
        Cell& operator*() {
            return cell_;
        }

    private:
        std::unique_lock<std::mutex> lock_;
        Cell& cell_;
    };
    CellGuard get_cell(int idx) noexcept {
        return CellGuard(voxel_map_->grid[idx], *cell_mutexes_[idx]);
    }
    void reset_a_cell(int idx) noexcept {
        auto cell = get_cell(idx);
        cell.get().reset();
    }

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
    LogCtx log_ctx_;
    auto& get_log_ctx() {
        return log_ctx_;
    }
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
        void _reset() {
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
    void track_state(int idx) noexcept {
        int p = occupied_pos_[idx];
        bool was_occ = (p != -1);
        bool now_occ = is_occupied(idx);

        if (was_occ && !now_occ) {
            std::lock_guard<std::mutex> lock(occupied_mutex_);
            int last = occupied_buffer_idx_.back();
            occupied_buffer_idx_[p] = last;
            occupied_pos_[last] = p;
            occupied_buffer_idx_.pop_back();
            occupied_pos_[idx] = -1;
        } else if (!was_occ && now_occ) {
            std::lock_guard<std::mutex> lock(occupied_mutex_);
            occupied_pos_[idx] = occupied_buffer_idx_.size();
            occupied_buffer_idx_.push_back(idx);
        }
    }
    std::vector<int> get_occupied_idx() const noexcept {
        std::lock_guard<std::mutex> lock(occupied_mutex_);
        return occupied_buffer_idx_;
    }
    SlidingVoxelMap<3, Cell>::Ptr get_voxel_map() const noexcept {
        return voxel_map_;
    }
    SlidingVoxelMap<3, Cell>::Ptr voxel_map_;
    std::vector<std::unique_ptr<std::mutex>> cell_mutexes_;
    mutable std::mutex occupied_mutex_;
    std::vector<int> occupied_buffer_idx_;
    std::vector<int> occupied_pos_;
    double now_;
    LockQueue<IdxSnap<double>> free_queue_;
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
    std::thread receive_thread_, free_thread_, ray_thread_;
};
OccMap::OccMap(const ParamsNode& config) {
    _impl = std::make_unique<Impl>(config);
}
OccMap::~OccMap() {
    _impl.reset();
}
void OccMap::insert_point_cloud(OccMap::Frame&& frame) noexcept {
    _impl->insert_point_cloud(std::move(frame));
}
inline bool OccMap::is_occupied(int idx) const noexcept {
    return _impl->is_occupied(idx);
}

void OccMap::update(double now) noexcept {
    _impl->update(now);
}
Eigen::Vector3f OccMap::center() const {
    return _impl->center();
}
std::vector<Eigen::Vector4f> OccMap::get_occupied_points() const noexcept {
    return _impl->get_occupied_points();
}

void OccMap::set_center(const Eigen::Vector3f& o) noexcept {
    _impl->set_center(o);
}

inline int OccMap::key_to_index(const VoxelKey<3>& k) const noexcept {
    return _impl->key_to_index(k);
}
inline VoxelKey<3> OccMap::index_to_key(int idx) const noexcept {
    return _impl->index_to_key(idx);
}

inline VoxelKey<3> OccMap::world_to_key(const Eigen::Vector3f& p) const noexcept {
    return _impl->world_to_key(p);
}
inline Eigen::Vector3f OccMap::key_to_world(const VoxelKey<3>& k) const noexcept {
    return _impl->key_to_world(k);
}
OccMap::LogCtx& OccMap::get_log_ctx() {
    return _impl->get_log_ctx();
}
std::vector<int> OccMap::get_occupied_idx() const noexcept {
    return _impl->get_occupied_idx();
}
SlidingVoxelMap<3, OccMap::Cell>::Ptr OccMap::get_voxel_map() const noexcept {
    return _impl->get_voxel_map();
}
} // namespace rose_nav::map
