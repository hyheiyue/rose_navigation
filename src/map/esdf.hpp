#pragma once
#include "bin_map.hpp"

#include <cmath>

namespace rose_nav::map {

class ESDF {
public:
    using Ptr = std::shared_ptr<ESDF>;

    explicit ESDF(BinMap::Ptr bin, const ParamsNode& config);

    static Ptr create(BinMap::Ptr bin, const ParamsNode& config) {
        return std::make_shared<ESDF>(bin, config);
    }

    void update();
    void set_center(const Eigen::Vector2f& o) {
        if (bin_map_->has_static_) {
            return;
        }
        VoxelKey<2> new_center = esdf_->world_to_key(o);
        VoxelKey<2> shift { new_center.x() - esdf_->center_key.x(),
                            new_center.y() - esdf_->center_key.y() };
        int min_shift = 1;
        if (std::abs(shift.x()) < min_shift && std::abs(shift.y()) < min_shift)
            return;
        esdf_->slide_to(new_center, [&](int idx) {

        });
    }
    float get_esdf(const Eigen::Vector2f& p) {
        int idx = esdf_->world_to_index(p);
        if (idx >= 0) {
            return esdf_->grid[idx];
        }
        return kInf;
    }
    float get_esdf(int i) {
        if (i >= 0) {
            return esdf_->grid[i];
        }
        return kInf;
    }
    inline int world_to_index(const Eigen::Vector2f& p) const noexcept {
        return esdf_->world_to_index(p);
    }

    inline VoxelKey<2> world_to_key(const Eigen::Vector2f& p) const noexcept {
        return esdf_->world_to_key(p);
    }

    inline Eigen::Vector2f key_to_world(const VoxelKey<2>& k) const noexcept {
        return esdf_->key_to_world(k);
    }

    inline Eigen::Vector2f index_to_world(int idx) const noexcept {
        return esdf_->index_to_world(idx);
    }

    inline int key_to_index(const VoxelKey<2>& k) const noexcept {
        return esdf_->key_to_index(k);
    }

    inline VoxelKey<2> index_to_key(int idx) const noexcept {
        return esdf_->index_to_key(idx);
    }
    std::vector<Eigen::Vector4f> get_occupied_points(int step = 1) const;

    static constexpr float kInf = 1e6f;
    SlidingVoxelMap<2, float>::Ptr esdf_;
    std::vector<float> dist_to_occ_;
    std::vector<float> dist_to_free_;
    BinMap::Ptr bin_map_;
    static constexpr int dx8_[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static constexpr int dy8_[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

    float step_cost_[2];

    inline bool is_diagonal_idx(int k) const {
        return k >= 4;
    }

    void propagate_key_distance_field_two_pass(
        const std::vector<uint8_t>& acc,
        std::vector<float>& dist,
        bool source_is_occ
    );

    void rebuild_signed();
};

} // namespace rose_nav::map
