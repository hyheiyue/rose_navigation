#pragma once
#include "bin_map.hpp"
#include "esdf.hpp"
#include "occ_map.hpp"

namespace rose_nav::map {

class RoseMap {
public:
    using Ptr = std::shared_ptr<RoseMap>;
    RoseMap(rclcpp::Node& node);
    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<RoseMap>(node);
    }
    ~RoseMap();
    // 三层地图接口：OccMap 保存 3D 概率占据，BinMap 投影为 2D 可通行性，
    // ESDF 在 BinMap 上计算到障碍物的有符号距离，供规划器快速查询安全距离。
    OccMap::Ptr occ_map() const;
    BinMap::Ptr bin_map() const;
    ESDF::Ptr esdf() const;
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rose_nav::map
