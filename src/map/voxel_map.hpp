#pragma once

#include "ankerl/unordered_dense.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>
namespace rose_nav::map {

template<int Dim>
struct VoxelKey {
    std::array<int, Dim> data {};

    int& operator[](int i) noexcept {
        return data[i];
    }
    const int& operator[](int i) const noexcept {
        return data[i];
    }

    // ----- x -----
    int& x() noexcept {
        static_assert(Dim >= 1, "x requires Dim >= 1");
        return data[0];
    }
    const int& x() const noexcept {
        static_assert(Dim >= 1, "x requires Dim >= 1");
        return data[0];
    }

    // ----- y -----
    int& y() noexcept {
        static_assert(Dim >= 2, "y requires Dim >= 2");
        return data[1];
    }
    const int& y() const noexcept {
        static_assert(Dim >= 2, "y requires Dim >= 2");
        return data[1];
    }

    // ----- z -----
    int& z() noexcept {
        static_assert(Dim >= 3, "z requires Dim >= 3");
        return data[2];
    }
    const int& z() const noexcept {
        static_assert(Dim >= 3, "z requires Dim >= 3");
        return data[2];
    }
    bool operator==(const VoxelKey& other) const noexcept {
        for (int i = 0; i < Dim; ++i)
            if (data[i] != other.data[i])
                return false;
        return true;
    }

    bool operator!=(const VoxelKey& other) const noexcept {
        return !(*this == other);
    }

    VoxelKey& operator+=(const VoxelKey& other) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] += other.data[i];
        return *this;
    }

    VoxelKey& operator-=(const VoxelKey& other) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    VoxelKey& operator*=(int scalar) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] *= scalar;
        return *this;
    }

    VoxelKey& operator/=(int scalar) noexcept {
        for (int i = 0; i < Dim; ++i)
            data[i] /= scalar;
        return *this;
    }

    friend VoxelKey operator+(VoxelKey lhs, const VoxelKey& rhs) noexcept {
        lhs += rhs;
        return lhs;
    }

    friend VoxelKey operator-(VoxelKey lhs, const VoxelKey& rhs) noexcept {
        lhs -= rhs;
        return lhs;
    }

    friend VoxelKey operator*(VoxelKey lhs, int scalar) noexcept {
        lhs *= scalar;
        return lhs;
    }

    friend VoxelKey operator*(int scalar, VoxelKey rhs) noexcept {
        rhs *= scalar;
        return rhs;
    }

    friend VoxelKey operator/(VoxelKey lhs, int scalar) noexcept {
        lhs /= scalar;
        return lhs;
    }
    constexpr VoxelKey cwise_min(const VoxelKey& other) const noexcept {
        VoxelKey out {};
        for (int i = 0; i < Dim; ++i)
            out.data[i] = data[i] < other.data[i] ? data[i] : other.data[i];
        return out;
    }

    constexpr VoxelKey cwise_max(const VoxelKey& other) const noexcept {
        VoxelKey out {};
        for (int i = 0; i < Dim; ++i)
            out.data[i] = data[i] > other.data[i] ? data[i] : other.data[i];
        return out;
    }
};

template<int Dim, typename Cell>
class HashVoxelMap {
    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

public:
    using Ptr = std::shared_ptr<HashVoxelMap>;
    using Key = VoxelKey<Dim>;
    using EigenPoint = Eigen::Matrix<float, Dim, 1>;

    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            uint64_t x = static_cast<uint32_t>(k.x());
            uint64_t y = static_cast<uint32_t>(k.y());

            if constexpr (Dim == 2) {
                uint64_t h = x * 0x9E3779B185EBCA87ull ^ y * 0xC2B2AE3D27D4EB4Full;
                return final_mix(h);
            } else {
                uint64_t z = static_cast<uint32_t>(k.z());

                uint64_t h = x * 0x9E3779B185EBCA87ull ^ y * 0xC2B2AE3D27D4EB4Full
                    ^ z * 0x165667B19E3779F9ull;

                return final_mix(h);
            }
        }

    private:
        static inline uint64_t final_mix(uint64_t h) noexcept {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        }
    };

    explicit HashVoxelMap(float voxel_size_): voxel_size(voxel_size_) {}

    static Ptr create(float voxel_size) {
        return std::make_shared<HashVoxelMap>(voxel_size);
    }

    inline Key world_to_key(const EigenPoint& p) const noexcept {
        Key k;
        const float inv = 1.0f / voxel_size;
        for (int i = 0; i < Dim; ++i) {
            k.data[i] = static_cast<int>(std::floor(p[i] * inv + 1e-6f));
        }
        return k;
    }

    inline EigenPoint key_to_world(const Key& k) const noexcept {
        EigenPoint p;
        for (int i = 0; i < Dim; ++i) {
            p[i] = (k.data[i] + 0.5f) * voxel_size;
        }
        return p;
    }

    void set_cell(const EigenPoint& pos, const Cell& value) {
        Key k = world_to_key(pos);

        grid[k] = value;

        if (grid.size() == 1) {
            min_key = k;
            max_key = k;
        } else {
            min_key = min_key.cwise_min(k);
            max_key = max_key.cwise_max(k);
        }
    }

    Cell* get_cell(const EigenPoint& pos) {
        Key k = world_to_key(pos);
        auto it = grid.find(k);
        if (it != grid.end()) {
            return &it->second;
        }
        return nullptr;
    }

    void remove_cell(const EigenPoint& pos) {
        Key k = world_to_key(pos);
        auto it = grid.find(k);
        if (it == grid.end()) {
            return;
        }

        grid.erase(it);

        if (grid.empty()) {
            return;
        }

        bool touch_boundary = false;
        for (int i = 0; i < Dim; ++i) {
            if (k[i] == min_key[i] || k[i] == max_key[i]) {
                touch_boundary = true;
                break;
            }
        }

        if (touch_boundary) {
            recompute_min_max();
        }
    }

    void recompute_min_max() {
        auto it = grid.begin();
        min_key = it->first;
        max_key = it->first;

        ++it;
        for (; it != grid.end(); ++it) {
            const Key& k = it->first;
            min_key = min_key.cwise_min(k);
            max_key = max_key.cwise_max(k);
        }
    }

    void clear() {
        grid.clear();
    }

    size_t size() const {
        return grid.size();
    }

public:
    Key min_key;
    Key max_key;

    float voxel_size;

    ankerl::unordered_dense::map<Key, Cell, KeyHash> grid;
};
template<int Dim, typename Cell>
class SlidingVoxelMap {
    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

public:
    using Ptr = std::shared_ptr<SlidingVoxelMap>;
    using Key = VoxelKey<Dim>;
    using EigenPoint = Eigen::Matrix<float, Dim, 1>;

    SlidingVoxelMap(float voxel_size_, const EigenPoint& size_, const EigenPoint& center_):
        voxel_size(voxel_size_),
        size(size_),
        center(center_) {
        EigenPoint half = size * 0.5f;

        min_key = world_to_key(center - half);
        max_key = world_to_key(center + half);

        for (int i = 0; i < Dim; ++i) {
            dims[i] = max_key[i] - min_key[i] + 1;
            offset[i] = 0;
        }

        center_key = world_to_key(center);

        size_t N = 1;
        for (int i = 0; i < Dim; ++i)
            N *= static_cast<size_t>(dims[i]);

        grid.resize(N);
    }
    SlidingVoxelMap(
        float voxel_size_,
        const EigenPoint& min_pos,
        const EigenPoint& max_pos,
        bool /*dummy*/
    ):
        voxel_size(voxel_size_) {
        min_key = world_to_key(min_pos);
        max_key = world_to_key(max_pos);

        for (int i = 0; i < Dim; ++i) {
            if (max_key[i] < min_key[i])
                std::swap(max_key[i], min_key[i]);

            dims[i] = max_key[i] - min_key[i] + 1;
            offset[i] = 0;
        }

        for (int i = 0; i < Dim; ++i)
            center_key[i] = (min_key[i] + max_key[i]) / 2;

        EigenPoint min_world = key_to_world(min_key);
        EigenPoint max_world = key_to_world(max_key);

        for (int i = 0; i < Dim; ++i) {
            center[i] = (min_world[i] + max_world[i]) * 0.5f;
            size[i] = dims[i] * voxel_size;
        }

        size_t N = 1;
        for (int i = 0; i < Dim; ++i)
            N *= static_cast<size_t>(dims[i]);

        grid.resize(N);
    }

    static Ptr create(float voxel_size, const EigenPoint& size, const EigenPoint& center) {
        return std::make_shared<SlidingVoxelMap>(voxel_size, size, center);
    }

    size_t grid_size() const noexcept {
        return grid.size();
    }
    inline int world_to_index(const EigenPoint& p) const noexcept {
        return key_to_index(world_to_key(p));
    }

    inline Key world_to_key(const EigenPoint& p) const noexcept {
        Key k;
        const float inv = 1.0f / voxel_size;
        for (int i = 0; i < Dim; ++i)
            k.data[i] = static_cast<int>(std::floor(p[i] * inv + 1e-6f));
        return k;
    }

    inline EigenPoint key_to_world(const Key& k) const noexcept {
        EigenPoint p;
        for (int i = 0; i < Dim; ++i)
            p[i] = (k.data[i] + 0.5f) * voxel_size;
        return p;
    }

    inline EigenPoint index_to_world(int idx) const noexcept {
        return key_to_world(index_to_key(idx));
    }

    inline int key_to_index(const Key& k) const noexcept {
        int idx = 0;
        int stride = 1;

        for (int d = Dim - 1; d >= 0; --d) {
            int delta = k[d] - center_key[d] + (dims[d] >> 1);

            if (delta < 0 || delta >= dims[d])
                return -1;

            int r = delta + offset[d];

            if (r >= dims[d])
                r -= dims[d];
            else if (r < 0)
                r += dims[d];

            idx += r * stride;
            stride *= dims[d];
        }

        return idx;
    }

    inline Key index_to_key(int idx) const noexcept {
        Key k;

        for (int d = Dim - 1; d >= 0; --d) {
            int r = idx % dims[d];
            idx /= dims[d];

            int delta = r - offset[d];

            if (delta < 0)
                delta += dims[d];
            else if (delta >= dims[d])
                delta -= dims[d];

            k[d] = center_key[d] + delta - (dims[d] >> 1);
        }

        return k;
    }
    template<typename ClearFunc>
    void slide_to(const Key& new_center_key, ClearFunc clear_func) {
        Key shift;
        for (int d = 0; d < Dim; ++d)
            shift[d] = new_center_key[d] - center_key[d];

        for (int d = 0; d < Dim; ++d) {
            if (std::abs(shift[d]) >= dims[d]) {
                for (size_t i = 0; i < grid.size(); ++i)
                    clear_func(i);

                offset = {};
                center_key = new_center_key;
                return;
            }
        }
        for (int axis = 0; axis < Dim; ++axis) {
            int s = shift[axis];
            if (s == 0)
                continue;

            int steps = std::abs(s);
            int dir = s > 0 ? 1 : -1;

            for (int step = 0; step < steps; ++step) {
                int slice = (offset[axis] + dir * step + dims[axis]) % dims[axis];

                clear_slice(axis, slice, clear_func);
            }

            offset[axis] = (offset[axis] + s + dims[axis]) % dims[axis];
        }

        center_key = new_center_key;
        EigenPoint half = size * 0.5f;
        center = key_to_world(center_key);
        min_key = world_to_key(center - half);
        max_key = world_to_key(center + half);
    }

    template<typename ClearFunc>
    void clear_slice(int axis, int slice, ClearFunc clear_func) {
        if constexpr (Dim == 3) {
            int dx = dims[0];
            int dy = dims[1];
            int dz = dims[2];

            if (axis == 0) {
                for (int y = 0; y < dy; ++y)
                    for (int z = 0; z < dz; ++z) {
                        int idx = (slice * dy + y) * dz + z;
                        clear_func(idx);
                    }
            } else if (axis == 1) {
                for (int x = 0; x < dx; ++x)
                    for (int z = 0; z < dz; ++z) {
                        int idx = (x * dy + slice) * dz + z;
                        clear_func(idx);
                    }
            } else {
                for (int x = 0; x < dx; ++x)
                    for (int y = 0; y < dy; ++y) {
                        int idx = (x * dy + y) * dz + slice;
                        clear_func(idx);
                    }
            }
        }
    }
    auto get_center() const noexcept {
        return center;
    }

public:
    float voxel_size;

    Key dims;
    Key offset;

    std::vector<Cell> grid;

    Key center_key;
    EigenPoint center;
    EigenPoint size;
    Key min_key;
    Key max_key;
};

} // namespace rose_nav::map