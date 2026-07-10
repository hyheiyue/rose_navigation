#pragma once

#include "ankerl/unordered_dense.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <utility>
#include <vector>

namespace rose_nav::lm {

template<int K, typename Point>
class KNNHeap {
public:
    KNNHeap() {
        reset();
    }

    void reset() {
        count = 0;
        worst_ = 0;
        max_dist2_ = 0.0f;
        std::memset(dist2_, 0, sizeof(dist2_));
    }

    void try_insert(float dist2, const Point& point) {
        const bool not_full = count < K;
        const bool should_insert = not_full || dist2 < max_dist2_;
        if (!should_insert) {
            return;
        }

        const uint8_t insert_idx = not_full ? count : worst_;
        dist2_[insert_idx] = dist2;
        points_[insert_idx] = point;

        if (not_full) {
            ++count;
            if (dist2 > max_dist2_) {
                max_dist2_ = dist2;
                worst_ = insert_idx;
            }
            return;
        }

        update_worst_unrolled();
    }

    [[nodiscard]] float max_dist2() const noexcept {
        return max_dist2_;
    }

    uint8_t count = 0;
    uint8_t worst_ = 0;
    float max_dist2_ = 0.0f;
    float dist2_[K] {};
    std::array<Point, K> points_ {};

private:
    void update_worst_unrolled() {
        static_assert(K == 5, "update_worst_unrolled is specialized for KNNHeap<5>");

        const float d0 = dist2_[0];
        const float d1 = dist2_[1];
        const float d2 = dist2_[2];
        const float d3 = dist2_[3];
        const float d4 = dist2_[4];

        const uint8_t idx01 = d0 > d1 ? 0 : 1;
        const float max01 = d0 > d1 ? d0 : d1;

        const uint8_t idx23 = d2 > d3 ? 2 : 3;
        const float max23 = d2 > d3 ? d2 : d3;

        const uint8_t idx0123 = max01 > max23 ? idx01 : idx23;
        const float max0123 = max01 > max23 ? max01 : max23;

        worst_ = max0123 > d4 ? idx0123 : 4;
        max_dist2_ = max0123 > d4 ? max0123 : d4;
    }
};

class SmallOctVox {
public:
    using Ptr = std::shared_ptr<SmallOctVox>;
    using Point = Eigen::Vector3f;
    using PositionIndex = Eigen::Matrix<int, 3, 1>;
    using KNNHeapType = KNNHeap<5, Point>;

    explicit SmallOctVox(float resolution, size_t capacity = 1000000):
        resolution_(resolution),
        inv_resolution_(1.0f / resolution),
        sub_resolution_(resolution * 0.5f),
        sub_inv_resolution_(2.0f / resolution),
        capacity_(capacity) {
        grids_.reserve(capacity_);

        const float scale = resolution_ / 0.5f;
        const float scale2 = scale * scale;
        for (size_t i = 0; i < orders_min_dis2_.size(); ++i) {
            group_min_dist2_[i] = orders_min_dis2_[i] * scale2;
        }
    }

    bool add_point(const Point& point) {
        const FineKey fine_key = point_to_fine_key(point);
        const Key key { floor_div2(fine_key.x), floor_div2(fine_key.y), floor_div2(fine_key.z) };
        const uint64_t hash_key = pack_key(key);
        const uint8_t local_idx = fine_key_to_local_idx(fine_key);

        auto it = grids_.find(hash_key);
        if (it == grids_.end()) {
            voxels_.emplace_front(key, OctVox(point, local_idx));
            grids_.emplace(hash_key, voxels_.begin());

            if (voxels_.size() >= capacity_) {
                grids_.erase(pack_key(voxels_.back().first));
                voxels_.pop_back();
            }
            return true;
        }

        const bool accepted = it->second->second.add_point(point, local_idx);
        voxels_.splice(voxels_.begin(), voxels_, it->second);
        return accepted;
    }

    void
    get_closest_point(const Point& point, std::vector<Point>& closest_points, size_t max_num = 5)
        const {
        closest_points.clear();
        if (max_num == 0 || voxels_.empty()) {
            return;
        }

        max_num = std::min<size_t>(max_num, 5);

        KNNHeapType top_k;
        const FineKey fine_key = point_to_fine_key(point);
        const Key key { floor_div2(fine_key.x), floor_div2(fine_key.y), floor_div2(fine_key.z) };
        const uint8_t local_idx = fine_key_to_local_idx(fine_key);

        const int sx = (fine_key.x & 1) == 0 ? 1 : -1;
        const int sy = (fine_key.y & 1) == 0 ? 1 : -1;
        const int sz = (fine_key.z & 1) == 0 ? 1 : -1;

        std::array<const OctVox*, 60> cached_voxels {};
        std::array<uint8_t, 60> cached {};

        auto voxel_at = [&](uint8_t neighbor_idx) -> const OctVox* {
            if (cached[neighbor_idx] == 0) {
                cached[neighbor_idx] = 1;
                const Key neighbor_key =
                    offset_key(key, neighbor_voxels_[neighbor_idx], sx, sy, sz);
                const auto it = grids_.find(pack_key(neighbor_key));
                cached_voxels[neighbor_idx] = it == grids_.end() ? nullptr : &it->second->second;
            }
            return cached_voxels[neighbor_idx];
        };

        for (size_t group_idx = 0; group_idx + 1 < flat_search_order_offsets_.size(); ++group_idx) {
            size_t order_idx = flat_search_order_offsets_[group_idx];
            const size_t group_end = flat_search_order_offsets_[group_idx + 1];

            while (order_idx < group_end) {
                const uint8_t neighbor_idx = flat_search_order_[order_idx++];
                uint8_t data_size = flat_search_order_[order_idx++];

                const OctVox* voxel = voxel_at(neighbor_idx);

                if (voxel == nullptr) {
                    order_idx += data_size;
                    continue;
                }

                while (data_size-- > 0) {
                    Point sub_point;
                    const uint8_t mirrored_idx = flat_search_order_[order_idx++] ^ local_idx;
                    if (!voxel->get_point(mirrored_idx, sub_point)) {
                        continue;
                    }
                    top_k.try_insert((sub_point - point).squaredNorm(), sub_point);
                }
            }

            if (top_k.count == 5 && top_k.max_dist2_ < group_min_dist2_[group_idx]) {
                break;
            }
        }

        const size_t keep_num = std::min(max_num, static_cast<size_t>(top_k.count));
        closest_points.reserve(keep_num);
        for (size_t i = 0; i < keep_num; ++i) {
            closest_points.push_back(top_k.points_[i]);
        }
    }

    void clear() {
        grids_.clear();
        voxels_.clear();
    }

    [[nodiscard]] size_t size() const noexcept {
        return voxels_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return voxels_.empty();
    }

    [[nodiscard]] float resolution() const noexcept {
        return resolution_;
    }

    [[nodiscard]] PositionIndex get_position_index(const Point& point) const noexcept {
        const FineKey key = point_to_fine_key(point);
        return PositionIndex(floor_div2(key.x), floor_div2(key.y), floor_div2(key.z));
    }

    void get_points(std::vector<Point>& points) const {
        points.clear();
        points.reserve(voxels_.size() * kSubVoxelNum);
        for (const auto& item: voxels_) {
            const OctVox& voxel = item.second;
            for (uint8_t i = 0; i < kSubVoxelNum; ++i) {
                Point point;
                if (voxel.get_point(i, point)) {
                    points.push_back(point);
                }
            }
        }
    }

private:
    struct Key {
        int x = 0;
        int y = 0;
        int z = 0;

        bool operator==(const Key& other) const noexcept {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct FineKey {
        int x = 0;
        int y = 0;
        int z = 0;
    };

    class OctVox {
    public:
        OctVox(const Point& point, uint8_t local_idx) {
            counts_.fill(0);
            points_[local_idx] = point;
            counts_[local_idx] = 1;
        }

        bool add_point(const Point& point, uint8_t local_idx) {
            uint8_t& count = counts_[local_idx];
            Point& stored_point = points_[local_idx];
            if (count == 0) {
                stored_point = point;
                count = 1;
                return true;
            }

            if (count >= kMaxPointsPerSubVoxel) {
                return false;
            }
            if ((point - stored_point).squaredNorm() > kFuseDistanceSquared) {
                return false;
            }

            stored_point =
                (stored_point * static_cast<float>(count) + point) / static_cast<float>(count + 1);
            ++count;
            return false;
        }

        bool get_point(uint8_t local_idx, Point& point) const {
            if (counts_[local_idx] == 0) {
                return false;
            }
            point = points_[local_idx];
            return true;
        }

    private:
        static constexpr uint8_t kMaxPointsPerSubVoxel = 20;
        static constexpr float kFuseDistanceSquared = 0.01f;

        std::array<uint8_t, 8> counts_ {};
        std::array<Point, 8> points_ {};
    };

    using VoxelList = std::list<std::pair<Key, OctVox>>;
    using VoxelIter = typename VoxelList::iterator;

    static constexpr uint8_t kSubVoxelNum = 8;

    static int floor_div2(int value) noexcept {
        return value >= 0 ? value / 2 : (value - 1) / 2;
    }

    FineKey point_to_fine_key(const Point& point) const noexcept {
        return {
            static_cast<int>(std::floor(point.x() * sub_inv_resolution_)),
            static_cast<int>(std::floor(point.y() * sub_inv_resolution_)),
            static_cast<int>(std::floor(point.z() * sub_inv_resolution_)),
        };
    }

    static uint8_t fine_key_to_local_idx(const FineKey& key) noexcept {
        const uint8_t dx = static_cast<uint8_t>(key.x & 1);
        const uint8_t dy = static_cast<uint8_t>(key.y & 1);
        const uint8_t dz = static_cast<uint8_t>(key.z & 1);
        return static_cast<uint8_t>((dz << 2) | (dy << 1) | dx);
    }

    static Key offset_key(const Key& base, const Key& offset, int sx, int sy, int sz) noexcept {
        return {
            base.x + sx * offset.x,
            base.y + sy * offset.y,
            base.z + sz * offset.z,
        };
    }

    static uint64_t pack_key(const Key& key) noexcept {
        constexpr uint64_t mask = (uint64_t { 1 } << 21) - 1;
        return (static_cast<uint64_t>(key.x) & mask) << 42
            | (static_cast<uint64_t>(key.y) & mask) << 21 | (static_cast<uint64_t>(key.z) & mask);
    }

    float resolution_ = 0.5f;
    float inv_resolution_ = 2.0f;
    float sub_resolution_ = 0.25f;
    float sub_inv_resolution_ = 4.0f;
    size_t capacity_ = 1000000;
    std::array<float, 6> group_min_dist2_ {};

    VoxelList voxels_;
    ankerl::unordered_dense::map<uint64_t, VoxelIter> grids_;

    static constexpr std::array<float, 6> orders_min_dis2_ = {
        0.062500f, 0.125000f, 0.250000f, 0.312500f, 0.375000f, 100.0f,
    };

    static constexpr std::array<uint16_t, 7> flat_search_order_offsets_ = {
        0, 43, 135, 219, 321, 465, 593,
    };

    static constexpr std::array<Key, 60> neighbor_voxels_ = {
        Key { 0, 0, 0 },    Key { 0, -1, 0 },   Key { 0, 0, -1 },   Key { -1, 0, 0 },
        Key { -1, 0, -1 },  Key { 0, -1, -1 },  Key { -1, -1, 0 },  Key { -1, -1, -1 },
        Key { 1, 0, 0 },    Key { 0, 1, 0 },    Key { 0, 0, 1 },    Key { 0, -1, 1 },
        Key { 1, -1, 0 },   Key { 1, 0, -1 },   Key { 0, 1, -1 },   Key { -1, 1, 0 },
        Key { -1, 0, 1 },   Key { -1, -1, 1 },  Key { -1, 1, -1 },  Key { 1, -1, -1 },
        Key { 0, 0, -2 },   Key { 0, 1, 1 },    Key { 1, 1, 0 },    Key { 1, 0, 1 },
        Key { -2, 0, 0 },   Key { 0, -2, 0 },   Key { -2, 0, -1 },  Key { -1, 0, -2 },
        Key { -1, -2, 0 },  Key { -2, -1, 0 },  Key { 0, -2, -1 },  Key { 1, 1, -1 },
        Key { -1, 1, 1 },   Key { 1, -1, 1 },   Key { 0, -1, -2 },  Key { -2, -1, -1 },
        Key { -1, -1, -2 }, Key { -1, -2, -1 }, Key { -2, 1, 0 },   Key { 1, -2, 0 },
        Key { 0, -2, 1 },   Key { 0, 1, -2 },   Key { 1, 0, -2 },   Key { 1, 1, 1 },
        Key { -2, 0, 1 },   Key { 1, -1, -2 },  Key { 1, -2, -1 },  Key { -2, 1, -1 },
        Key { -2, -1, 1 },  Key { -1, 1, -2 },  Key { -1, -2, 1 },  Key { 0, -2, -2 },
        Key { -2, 0, -2 },  Key { -2, 1, 1 },   Key { 1, 1, -2 },   Key { 1, -2, 1 },
        Key { -2, -2, 0 },  Key { -2, -1, -2 }, Key { -2, -2, -1 }, Key { -1, -2, -2 },
    };

    static constexpr std::array<uint8_t, 593> flat_search_order_ = {
        0,  8,  0,  1,  2,  3,  4,  5,  6,  7,  1,  4,  2,  3,  6,  7,  2,  4,  4,  5,  6,  7,  3,
        4,  1,  3,  5,  7,  4,  2,  5,  7,  5,  2,  6,  7,  6,  2,  3,  7,  7,  1,  7,

        1,  4,  0,  1,  4,  5,  2,  4,  0,  1,  2,  3,  3,  4,  0,  2,  4,  6,  4,  4,  1,  3,  4,
        6,  5,  4,  2,  3,  4,  5,  6,  4,  1,  2,  5,  6,  7,  3,  3,  5,  6,  8,  4,  0,  2,  4,
        6,  9,  4,  0,  1,  4,  5,  10, 4,  0,  1,  2,  3,  11, 2,  2,  3,  12, 2,  2,  6,  13, 2,
        4,  6,  14, 2,  4,  5,  15, 2,  1,  5,  16, 2,  1,  3,  17, 1,  3,  18, 1,  5,  19, 1,  6,

        4,  2,  0,  2,  5,  2,  0,  1,  6,  2,  0,  4,  7,  4,  0,  1,  2,  4,  11, 2,  0,  1,  12,
        2,  0,  4,  13, 2,  0,  2,  14, 2,  0,  1,  15, 2,  0,  4,  16, 2,  0,  2,  17, 3,  0,  1,
        2,  18, 3,  0,  1,  4,  19, 3,  0,  2,  4,  21, 2,  0,  1,  22, 2,  0,  4,  23, 2,  0,  2,
        31, 2,  0,  4,  32, 2,  0,  1,  33, 2,  0,  2,  43, 1,  0,

        8,  4,  1,  3,  5,  7,  9,  4,  2,  3,  6,  7,  10, 4,  4,  5,  6,  7,  11, 2,  6,  7,  12,
        2,  3,  7,  13, 2,  5,  7,  14, 2,  6,  7,  15, 2,  3,  7,  16, 2,  5,  7,  17, 1,  7,  18,
        1,  7,  19, 1,  7,  20, 4,  4,  5,  6,  7,  24, 4,  1,  3,  5,  7,  25, 4,  2,  3,  6,  7,
        26, 2,  5,  7,  27, 2,  5,  7,  28, 2,  3,  7,  29, 2,  3,  7,  30, 2,  6,  7,  34, 2,  6,
        7,  35, 1,  7,  36, 1,  7,  37, 1,  7,

        11, 2,  4,  5,  12, 2,  1,  5,  13, 2,  1,  3,  14, 2,  2,  3,  15, 2,  2,  6,  16, 2,  4,
        6,  17, 2,  5,  6,  18, 2,  3,  6,  19, 2,  3,  5,  21, 4,  2,  3,  4,  5,  22, 4,  1,  2,
        5,  6,  23, 4,  1,  3,  4,  6,  26, 2,  1,  3,  27, 2,  4,  6,  28, 2,  2,  6,  29, 2,  1,
        5,  30, 2,  2,  3,  31, 2,  5,  6,  32, 2,  3,  5,  33, 2,  3,  6,  34, 2,  4,  5,  35, 2,
        3,  5,  36, 2,  5,  6,  37, 2,  3,  6,  38, 2,  1,  5,  39, 2,  2,  6,  40, 2,  2,  3,  41,
        2,  4,  5,  42, 2,  4,  6,  44, 2,  1,  3,  45, 1,  6,  46, 1,  6,  47, 1,  5,  48, 1,  3,
        49, 1,  5,  50, 1,  3,

        17, 1,  4,  18, 1,  2,  19, 1,  1,  21, 2,  6,  7,  22, 2,  3,  7,  23, 2,  5,  7,  31, 3,
        1,  2,  7,  32, 3,  2,  4,  7,  33, 3,  1,  4,  7,  35, 1,  1,  36, 1,  4,  37, 1,  2,  38,
        2,  3,  7,  39, 2,  3,  7,  40, 2,  6,  7,  41, 2,  6,  7,  42, 2,  5,  7,  43, 3,  1,  2,
        4,  44, 2,  5,  7,  45, 2,  4,  7,  46, 2,  2,  7,  47, 2,  1,  7,  48, 2,  1,  7,  49, 2,
        4,  7,  50, 2,  2,  7,  51, 2,  6,  7,  52, 2,  5,  7,  53, 1,  1,  54, 1,  4,  55, 1,  2,
        56, 2,  3,  7,  57, 1,  7,  58, 1,  7,  59, 1,  7,
    };
};

} // namespace rose_nav::lm
