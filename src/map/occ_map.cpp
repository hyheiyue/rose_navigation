#include "occ_map.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <thread>
#include <utility>
#include <vector>
namespace rose_nav::map {

OccMap::OccMap(const ParamsNode& config) {
    params_.load(config);
    auto voxel_map_config = config.sub("voxel_map");
    float voxel_size = voxel_map_config.declare<double>("voxel_size");
    auto size_vec = voxel_map_config.declare<std::vector<double>>("size");
    auto size = Eigen::Vector3f(size_vec[0], size_vec[1], size_vec[2]);
    auto center_vec = voxel_map_config.declare<std::vector<double>>("center");
    auto center = Eigen::Vector3f(center_vec[0], center_vec[1], center_vec[2]);
    voxel_map_ = SlidingVoxelMap<3, Cell>::create(voxel_size, size, center);
    const auto N = voxel_map_->grid_size();
    cell_mutexes.resize(N);
    for (auto& m: cell_mutexes) {
        m = std::make_unique<std::mutex>();
    }
    occupied_buffer_idx_.reserve(N);
    receive_thread_ = std::thread(std::bind(&OccMap::receive, this));
    hit_thread_ = std::thread(std::bind(&OccMap::hit, this));
    free_thread_ = std::thread(std::bind(&OccMap::free, this));
    if (params_.use_ray) {
        ray_thread_ = std::thread(std::bind(&OccMap::ray, this));
    }
}

void OccMap::insert_point_cloud(Frame&& frame) noexcept {
    frame_queue_.push(std::move(frame));
}
void OccMap::receive() noexcept {
    const auto N = voxel_map_->grid_size();
    IdxBuf hit(N);
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
        hit.indices.clear();
        hit.indices.reserve(pts.size());
        Ray ray;
        if (ray_buf) {
            ray_buf->indices.clear();
            ray_buf->indices.reserve(pts.size());
            ray.outmap.reserve(pts.size());
        }
        for (const auto& pt: pts) {
            const VoxelKey<3> k_hit = world_to_key(pt);

            int idx = key_to_index(k_hit);
            if (idx >= 0) {
                hit.try_push(idx, stamp);
            }
            if (ray_buf) {
                if (idx >= 0) {
                    ray_buf->try_push(idx, stamp);
                } else {
                    ray.outmap.push_back(k_hit);
                }
            }
        }
        hit_queue_.push(std::move(hit.make_snap(frame.time)));
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
void OccMap::hit() noexcept {
    while (rclcpp::ok()) {
        IdxSnap<double> hit;
        hit_queue_.wait_and_pop(hit);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < hit.indices.size(); i++) {
            int idx = hit.indices[i];
            Cell& c = voxel_map_->grid[idx];
            std::lock_guard<std::mutex> lock(*cell_mutexes[idx]);
            c.log_odds = std::max(c.log_odds + params_.log_hit * hit.count[i], params_.log_min);
            c.last_update = hit.ctx;
        }
        auto end = std::chrono::steady_clock::now();
        log_ctx_.hit_cost += std::chrono::duration<double, std::milli>(end - start).count();
        log_ctx_.hit_count += hit.indices.size();
    }
}
void OccMap::free() noexcept {
    while (rclcpp::ok()) {
        IdxSnap<double> free;
        free_queue_.wait_and_pop(free);
        auto start = std::chrono::steady_clock::now();
        tbb::parallel_for(
            size_t(0),
            free.indices.size(),
            [&](size_t i) {
                int idx = free.indices[i];
                Cell& c = voxel_map_->grid[idx];

                std::lock_guard<std::mutex> lock(*cell_mutexes[idx]);

                c.log_odds =
                    std::min(c.log_odds + params_.log_free * free.count[i], params_.log_max);
                c.last_update = free.ctx;
            },
            tbb::auto_partitioner()
        );
        auto end = std::chrono::steady_clock::now();
        log_ctx_.free_cost += std::chrono::duration<double, std::milli>(end - start).count();
        log_ctx_.free_count += free.indices.size();
    }
}
void OccMap::ray() noexcept {
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

        free.indices.clear();
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
void OccMap::update(double now) noexcept {
    now_ = now;
    auto& grid = voxel_map_->grid;
    const double timeout = params_.timeout;
    const auto N = grid.size();

    occupied_buffer_idx_.clear();
    occupied_buffer_idx_.reserve(N / 8);

    tbb::enumerable_thread_specific<std::vector<int>> tls([&] {
        std::vector<int> v;
        v.reserve(256);
        return v;
    });
    tbb::parallel_for(
        tbb::blocked_range<int>(0, N),
        [&](const tbb::blocked_range<int>& r) {
            auto& local = tls.local();
            for (int idx = r.begin(); idx != r.end(); ++idx) {
                if (is_occupied(idx)) {
                    local.push_back(idx);
                }
            }
        },
        tbb::auto_partitioner()
    );
    for (auto& local: tls) {
        occupied_buffer_idx_.insert(occupied_buffer_idx_.end(), local.begin(), local.end());
    }
}

std::vector<Eigen::Vector4f> OccMap::get_occupied_points(float sample_resolution_m) const noexcept {
    std::vector<Eigen::Vector4f> pts;

    if (sample_resolution_m <= 0.0f)
        sample_resolution_m = voxel_map_->voxel_size;

    int stride =
        std::max(1, static_cast<int>(std::round(sample_resolution_m / voxel_map_->voxel_size)));

    pts.reserve(occupied_buffer_idx_.size() / stride + 1);

    for (size_t i = 0; i < occupied_buffer_idx_.size(); i += stride) {
        int idx = occupied_buffer_idx_[i];
        if (!is_occupied(idx))
            continue;
        auto p = key_to_world(index_to_key(idx));
        pts.emplace_back(p.x(), p.y(), p.z(), p.z() - voxel_map_->center.z());
    }

    return pts;
}

void OccMap::set_center(const Eigen::Vector3f& o) noexcept {
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

} // namespace rose_nav::map
