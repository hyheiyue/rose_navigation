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
    OccMap::Ptr occ_map() const;
    BinMap::Ptr bin_map() const;
    ESDF::Ptr esdf() const;
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rose_nav::map