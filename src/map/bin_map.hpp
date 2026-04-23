#pragma once
#include "occ_map.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include "voxel_map.hpp"
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <yaml-cpp/yaml.h>
namespace rose_nav::map {
class BinMap {
public:
    using Ptr = std::shared_ptr<BinMap>;
    BinMap(const ParamsNode& config) {
        auto voxel_size = config.declare<float>("voxel_size");
        voxel_map_ = std::make_shared<HashVoxelMap<2, Cell>>(voxel_size);
        auto static_map_path = config.declare<std::string>("static_map_path");
        has_static_ = load_ros_map_yaml(static_map_path);
    }
    static Ptr create(const ParamsNode& config) {
        return std::make_shared<BinMap>(config);
    }
    template<typename UpdateFunc>
    void update(const UpdateFunc& update_func) {
        update_func(this);
    }
    std::vector<Eigen::Vector4f> get_occupied_points() const {
        std::vector<Eigen::Vector4f> pts;
        for (const auto& [key, cell]: voxel_map_->grid) {
            auto p = voxel_map_->key_to_world(key);
            pts.emplace_back(Eigen::Vector4f(p.x(), p.y(), 0, 1));
        }
        return pts;
    }
    bool load_ros_map_yaml(const std::string& yaml_path) {
        YAML::Node yaml;
        try {
            yaml = YAML::LoadFile(yaml_path);
        } catch (...) {
            RCLCPP_INFO_STREAM(
                rclcpp::get_logger("rose_nav:map"),
                "Failed to load yaml file: " << yaml_path
            );
            return false;
        }

        if (!yaml || !yaml.IsMap())
            return false;

        if (!yaml["image"] || !yaml["resolution"] || !yaml["origin"])
            return false;

        std::string image_path;
        float resolution;
        Eigen::Vector2f origin;
        try {
            image_path = yaml["image"].as<std::string>();
            resolution = yaml["resolution"].as<float>();

            const auto& origin_seq = yaml["origin"];
            if (!origin_seq.IsSequence() || origin_seq.size() < 2)
                return false;

            origin << origin_seq[0].as<float>(), origin_seq[1].as<float>();
        } catch (...) {
            return false;
        }

        if (!image_path.empty() && image_path[0] != '/') {
            const auto pos = yaml_path.find_last_of("/\\");
            if (pos != std::string::npos)
                image_path = yaml_path.substr(0, pos + 1) + image_path;
        }

        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty() || resolution <= 0.f)
            return false;
        Eigen::Vector2f size;
        size << img.cols, img.rows;
        static_min_pos_ = origin;

        static_max_pos_.x() = origin.x() + size.x() * resolution;
        static_max_pos_.y() = origin.y() + size.y() * resolution;
        const bool negate = yaml["negate"].as<int>(0) != 0;
        const float free_th = yaml["free_thresh"].as<float>(0.196f);
        const float occ_th = yaml["occupied_thresh"].as<float>(0.65f);

        voxel_map_->clear();

        const int width = img.cols;
        const int height = img.rows;

        for (int y = 0; y < height; ++y) {
            const uint8_t* row_ptr = img.ptr<uint8_t>(y);

            for (int x = 0; x < width; ++x) {
                uint8_t v = row_ptr[x];

                if (negate)
                    v = 255 - v;

                float occ = (255.f - v) / 255.f;

                if (occ < occ_th)
                    continue;

                float wx = origin.x() + (x + 0.5f) * resolution;
                float wy = origin.y() + (height - 1 - y + 0.5f) * resolution;

                voxel_map_->set_cell({ wx, wy }, Cell { .is_static = true });
            }
        }

        return true;
    }

    struct Cell {
        uint8_t is_static = false;
    };
    HashVoxelMap<2, Cell>::Ptr voxel_map_;
    bool has_static_ = false;
    Eigen::Vector2f static_min_pos_, static_max_pos_;
};
} // namespace rose_nav::map