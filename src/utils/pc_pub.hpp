#pragma once

#include "utils/utils.hpp"
#include <Eigen/Core>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <small_gicp/points/point_cloud.hpp>

namespace rose_nav {

class PcPub {
public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using Publisher = rclcpp::Publisher<PointCloudMsg>::SharedPtr;

    void create_a_lot(rclcpp::Node& node, const std::vector<std::string>& topics) noexcept {
        for (const auto& topic: topics) {
            pub_map_[topic] = node.create_publisher<PointCloudMsg>(topic, 10);
        }
    }
    bool topic_subscribed(const std::string& topic) noexcept {
        auto it = pub_map_.find(topic);
        if (it == pub_map_.end())
            return false;

        auto& pub = it->second;
        if (!utils::publisher_sub(pub))
            return false;
        return true;
    }

    template<typename CloudT>
    void publish(
        const CloudT& pts,
        PointCloudMsg& msg,
        const std::string& topic,
        int step = 1
    ) noexcept {
        auto it = pub_map_.find(topic);
        if (it == pub_map_.end())
            return;

        auto& pub = it->second;
        if (!utils::publisher_sub(pub))
            return;

        fill_and_publish(pts, msg, pub, step);
    }

private:
    void fill_and_publish(
        const std::vector<Eigen::Vector3f>& pts,
        PointCloudMsg& msg,
        const Publisher& pub,
        int step
    ) noexcept {
        const size_t real_size = pts.size() / step;
        setup_xyz(msg, real_size);

        for (size_t i = 0, j = 0; i < pts.size(); i += step, ++j) {
            float* ptr = reinterpret_cast<float*>(&msg.data[j * msg.point_step]);
            ptr[0] = pts[i].x();
            ptr[1] = pts[i].y();
            ptr[2] = pts[i].z();
        }

        pub->publish(msg);
    }

    void fill_and_publish(
        const std::vector<Eigen::Vector4f>& pts,
        PointCloudMsg& msg,
        const Publisher& pub,
        int step
    ) noexcept {
        const size_t real_size = pts.size() / step;
        setup_xyzi(msg, real_size);

        for (size_t i = 0, j = 0; i < pts.size(); i += step, ++j) {
            float* ptr = reinterpret_cast<float*>(&msg.data[j * msg.point_step]);
            ptr[0] = pts[i].x();
            ptr[1] = pts[i].y();
            ptr[2] = pts[i].z();
            ptr[3] = pts[i].w();
        }

        pub->publish(msg);
    }

    void fill_and_publish(
        const small_gicp::PointCloud& pts,
        PointCloudMsg& msg,
        const Publisher& pub,
        int step
    ) noexcept {
        const size_t real_size = pts.size() / step;
        setup_xyzi(msg, real_size);

        for (size_t i = 0, j = 0; i < pts.size(); i += step, ++j) {
            float* ptr = reinterpret_cast<float*>(&msg.data[j * msg.point_step]);
            auto p = pts.point(i);
            ptr[0] = p.x();
            ptr[1] = p.y();
            ptr[2] = p.z();
            ptr[3] = p.w();
        }

        pub->publish(msg);
    }

    void setup_xyz(PointCloudMsg& msg, size_t width) noexcept {
        msg.height = 1;
        msg.width = static_cast<uint32_t>(width);

        msg.fields = { make_field("x", 0), make_field("y", 4), make_field("z", 8) };

        msg.is_bigendian = false;
        msg.point_step = 12;
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step);
    }

    void setup_xyzi(PointCloudMsg& msg, size_t width) noexcept {
        msg.height = 1;
        msg.width = static_cast<uint32_t>(width);

        msg.fields = { make_field("x", 0),
                       make_field("y", 4),
                       make_field("z", 8),
                       make_field("intensity", 12) };

        msg.is_bigendian = false;
        msg.point_step = 16;
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step);
    }

    static sensor_msgs::msg::PointField
    make_field(const std::string& name, uint32_t offset) noexcept {
        sensor_msgs::msg::PointField f;
        f.name = name;
        f.offset = offset;
        f.datatype = sensor_msgs::msg::PointField::FLOAT32;
        f.count = 1;
        return f;
    }

    std::map<std::string, Publisher> pub_map_;
};

} // namespace rose_nav