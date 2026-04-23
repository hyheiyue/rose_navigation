#include "esdf.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <queue>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <vector>

namespace rose_nav::map {
ESDF::ESDF(BinMap::Ptr bin, const ParamsNode& config) {
    bin_map_ = bin;

    if (!bin_map_->has_static_) {
        auto size_vec = config.declare<std::vector<double>>("size");
        auto size = Eigen::Vector2f(size_vec[0], size_vec[1]);
        esdf_ = std::make_shared<SlidingVoxelMap<2, float>>(
            bin->voxel_map_->voxel_size,
            size,
            Eigen::Vector2f::Zero()
        );
    } else {
        esdf_ = std::make_shared<SlidingVoxelMap<2, float>>(
            bin->voxel_map_->voxel_size,
            bin->static_min_pos_,
            bin->static_max_pos_,
            true
        );
    }

    const auto N = esdf_->grid_size();
    dist_to_occ_.resize(N);
    dist_to_free_.resize(N);

    const float vs = bin->voxel_map_->voxel_size;
    step_cost_[0] = vs; // straight
    step_cost_[1] = vs * std::sqrt(2.0f); // diagonal
}
void ESDF::update() {
    rebuild_signed();
}
std::vector<Eigen::Vector4f> ESDF::get_occupied_points(int step) const {
    std::vector<Eigen::Vector4f> pts;
    const int N = esdf_->grid_size();
    step = std::max(1, step);

    pts.reserve(N / step);
    for (int i = 0; i < N; i += step) {
        auto p = esdf_->index_to_world(i);
        auto f = esdf_->grid[i];
        pts.emplace_back(Eigen::Vector4f(p.x(), p.y(), 0.0, f));
    }
    return pts;
}

void ESDF::propagate_key_distance_field_two_pass(
    const std::vector<uint8_t>& acc,
    std::vector<float>& dist,
    bool source_is_occ
) {
    const int size = esdf_->grid_size();
    if (size == 0)
        return;

    tbb::parallel_for(0, size, [&](int i) {
        const bool is_occ = (acc[i] != 0);
        dist[i] = (is_occ == source_is_occ) ? 0.0f : kInf;
    });

    if (std::none_of(dist.begin(), dist.end(), [](float v) { return v == 0.0f; })) {
        std::fill(dist.begin(), dist.end(), kInf);
        return;
    }

    const int Kmax = 8;

    const int min_x = esdf_->min_key.x();
    const int min_y = esdf_->min_key.y();
    const int max_x = esdf_->max_key.x();
    const int max_y = esdf_->max_key.y();

    if (min_x > max_x || min_y > max_y)
        return;

    auto relax = [&](int x, int y, bool forward) {
        VoxelKey<2> key;
        key.x() = x;
        key.y() = y;

        int idx = esdf_->key_to_index(key);
        if (idx < 0 || idx >= size)
            return;

        float best = dist[idx];
        if (!std::isfinite(best))
            return;

        for (int k = 0; k < Kmax; ++k) {
            // forward: 只用左/上
            if (forward && (dx8_[k] > 0 || dy8_[k] > 0))
                continue;

            // backward: 只用右/下
            if (!forward && (dx8_[k] < 0 || dy8_[k] < 0))
                continue;

            VoxelKey<2> nb;
            nb.x() = x + dx8_[k];
            nb.y() = y + dy8_[k];

            int nb_idx = esdf_->key_to_index(nb);
            if (nb_idx < 0 || nb_idx >= size)
                continue;

            float nd = dist[nb_idx];
            if (!std::isfinite(nd))
                continue;

            float step = step_cost_[is_diagonal_idx(k) ? 1 : 0];
            best = std::min(best, nd + step);
        }

        dist[idx] = best;
    };

    for (int y = min_y; y <= max_y; ++y)
        for (int x = min_x; x <= max_x; ++x)
            relax(x, y, true);

    for (int y = max_y; y >= min_y; --y)
        for (int x = max_x; x >= min_x; --x)
            relax(x, y, false);
}
void ESDF::rebuild_signed() {
    std::vector<uint8_t> acc(esdf_->grid_size(), false);
    for (const auto& [key, cell]: bin_map_->voxel_map_->grid) {
        auto p = bin_map_->voxel_map_->key_to_world(key);
        int idx = esdf_->world_to_index(p);
        if (idx >= 0) {
            acc[idx] = true;
        }
    }
    tbb::parallel_invoke(
        [&]() { propagate_key_distance_field_two_pass(acc, dist_to_occ_, true); },
        [&]() { propagate_key_distance_field_two_pass(acc, dist_to_free_, false); }
    );

    tbb::parallel_for(
        tbb::blocked_range<int>(0, esdf_->grid_size()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                esdf_->grid[i] = dist_to_occ_[i] - dist_to_free_[i];
            }
        }
    );
}

} // namespace rose_nav::map