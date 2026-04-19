#include "rose_map.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "small_gicp/points/point_cloud.hpp"
#include "small_gicp/util/downsampling_tbb.hpp"
#include "small_gicp/util/normal_estimation_tbb.hpp"
#include "utils/pc_pub.hpp"
#include "utils/rcl_tf.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Geometry/Transform.h>
#include <deque>
#include <functional>
#include <memory>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <thread>
namespace rose_nav::map {
struct RoseMap::Impl {
    struct BinParams {
        float cos_thresh;
        double bottom_z_to_robo_z;
        double top_z_to_robo_z;
        int count_thresh;
        void load(const ParamsNode& config) {
            float max_slope_deg = config.declare<float>("max_slope_deg");
            cos_thresh = std::cos(max_slope_deg * M_PI / 180.0f);
            bottom_z_to_robo_z = config.declare<double>("bottom_z_to_robo_z");
            top_z_to_robo_z = config.declare<double>("top_z_to_robo_z");
            count_thresh = config.declare<int>("count_thresh");
        }
    } bin_params_;
    Impl(rclcpp::Node& node) {
        node_ = &node;
        tf_ = std::make_unique<RclTF>(node);
        RCLCPP_INFO_STREAM(node.get_logger(), "[RoseMap] Initializing...");
        auto config = ParamsNode(node, "rose_map");
        occ_map_ = OccMap::create(config.sub("occ_map"));
        auto bin_ph = config.sub("bin_map");
        bin_params_.load(bin_ph);
        int max_update_rate = config.declare<int>("max_update_rate");
        max_update_dt_ = 1.0 / max_update_rate;
        bin_map_ = BinMap::create(bin_ph);
        esdf_ = ESDF::create(bin_map_, config.sub("esdf"));
        sensor_frame_ = config.declare<std::string>("sensor_frame");
        std::string pointcloud_topic = config.declare<std::string>("pointcloud_topic");
        pointcloud_sub_ = node.create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic,
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg) {
                const double ros_time =
                    pc_msg->header.stamp.sec + pc_msg->header.stamp.nanosec * 1e-9;
                static double t_init = -1.0;
                if (t_init < 0.0)
                    t_init = ros_time;
                last_header_ = pc_msg->header;
                double frame_time = ros_time - t_init;
                static Eigen::Isometry3d msg_in_target = Eigen::Isometry3d::Identity();
                auto msg_in_target_opt = tf_->get_transform(
                    target_frame_,
                    pc_msg->header.frame_id,
                    pc_msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1)
                );
                if (msg_in_target_opt) {
                    msg_in_target = *msg_in_target_opt;
                }

                const size_t size = pc_msg->width * pc_msg->height;
                std::vector<Eigen::Vector3f> pts(size);

                sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pc_msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> iter_y(*pc_msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> iter_z(*pc_msg, "z");

                for (size_t i = 0; i < size; ++i) {
                    Eigen::Vector4f p(*iter_x, *iter_y, *iter_z, 1.0f);
                    p = msg_in_target.cast<float>() * p;
                    pts[i] = p.head<3>();

                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                }

                static Eigen::Isometry3d sensor_in_target = Eigen::Isometry3d::Identity();
                auto sensor_in_target_opt = tf_->get_transform(
                    target_frame_,
                    sensor_frame_,
                    pc_msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1)
                );
                if (sensor_in_target_opt) {
                    sensor_in_target = *sensor_in_target_opt;
                }

                current_time_ = frame_time;

                occ_map_->insert_point_cloud(std::move(OccMap::Frame {
                    .time = current_time_,
                    .pts = std::move(pts),
                    .sensor_origin = sensor_in_target.translation().cast<float>(),
                }));
            }
        );
        auto odom_topic = config.declare<std::string>("odometry_topic");
        odometry_sub_ = node.create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            rclcpp::SensorDataQoS(),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                const auto& odom = *msg;
                auto T = tf_->get_transform(target_frame_, msg->header.frame_id, msg->header.stamp);
                if (!T.has_value()) {
                    return;
                }
                Eigen::Vector4d point_in_odom =
                    Eigen::Vector4d { static_cast<double>(odom.pose.pose.position.x),
                                      static_cast<double>(odom.pose.pose.position.y),
                                      static_cast<double>(odom.pose.pose.position.z),
                                      1.0 };
                auto p_in_target = T.value() * point_in_odom;
                occ_map_->set_center(
                    Eigen::Vector3f(p_in_target.x(), p_in_target.y(), p_in_target.z())
                );
                esdf_->set_center(Eigen::Vector2f(p_in_target.x(), p_in_target.y()));
            }
        );
        target_frame_ = config.declare<std::string>("target_frame");
        pc_pub_.create_a_lot(
            node,
            {
                OCC_MAP_TOPIC,
                ACC_MAP_TOPIC,
                ESDF_MAP_TOPIC,
            }
        );
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.transient_local();
        qos.reliable();
        grid_map_pub_ = node.create_publisher<nav_msgs::msg::OccupancyGrid>("map", qos);
        add_static_ = node.create_service<std_srvs::srv::Trigger>(
            "add_static",
            std::bind(
                &RoseMap::Impl::add_static,
                this,
                std::placeholders::_1,
                std::placeholders::_2
            )
        );
        process_thread_ = std::thread(std::bind(&RoseMap::Impl::process_thread, this));
    }
    ~Impl() {
        if (process_thread_.joinable()) {
            process_thread_.join();
        }
    }

    static constexpr const char* OCC_MAP_TOPIC = "occ_map_out";
    static constexpr const char* ACC_MAP_TOPIC = "acc_map_out";
    static constexpr const char* ESDF_MAP_TOPIC = "esdf_out";

    void add_static(
        const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
        std_srvs::srv::Trigger::Response::SharedPtr /*res*/
    ) {
        auto& bin_voxel = bin_map_->voxel_map_;
        for (auto& hv: bin_voxel->grid) {
            auto& v = hv.second;
            v.is_static = true;
        }
        has_added_static_ = true;
    }
    void process_thread() {
        using Clock = std::chrono::steady_clock;

        auto next_tp = std::chrono::steady_clock::now();
        int update_count = 0;
        double update_cost = 0;
        while (rclcpp::ok()) {
            next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(max_update_dt_)
            );

            auto start = Clock::now();
            occ_map_->update(current_time_);
            bin_map_->update(std::bind(&RoseMap::Impl::bin_callback, this, std::placeholders::_1));
            esdf_->update();

            pub_all();

            auto end = Clock::now();
            update_count++;
            update_cost +=
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            utils::dt_once(
                [&]() {
                    auto& log_ctx = occ_map_->get_log_ctx();
                    RCLCPP_INFO_STREAM(
                        rclcpp::get_logger("rose_nav::map"),
                        "receive: " << log_ctx.receive_count << " cost: " << log_ctx.receive_cost
                                    << "ms"
                                    << " free: " << log_ctx.free_count
                                    << " cost: " << log_ctx.free_cost << "ms"
                                    << " hit: " << log_ctx.hit_count
                                    << " cost: " << log_ctx.hit_cost << "ms"
                                    << " ray: cost: " << log_ctx.ray_cost << "ms"
                                    << " up: " << update_count << " update: " << update_cost << "ms"
                    );
                    log_ctx.reset();
                    update_cost = 0;
                    update_count = 0;
                },
                std::chrono::duration<double>(1.0)
            );

            std::this_thread::sleep_until(next_tp);
        }
    }
    void pub_all() noexcept {
        if (pc_pub_.topic_subscribed(OCC_MAP_TOPIC)) {
            sensor_msgs::msg::PointCloud2 occ_msg;
            occ_msg.header.stamp = last_header_.stamp;
            occ_msg.header.frame_id = target_frame_;
            auto cloud = occ_map_->get_occupied_points();
            pc_pub_.publish(cloud, occ_msg, OCC_MAP_TOPIC);
        }

        if (pc_pub_.topic_subscribed(ACC_MAP_TOPIC)) {
            sensor_msgs::msg::PointCloud2 acc_msg;
            acc_msg.header.stamp = last_header_.stamp;
            acc_msg.header.frame_id = target_frame_;
            auto cloud = bin_map_->get_occupied_points();
            pc_pub_.publish(cloud, acc_msg, ACC_MAP_TOPIC);
        }

        if (pc_pub_.topic_subscribed(ESDF_MAP_TOPIC)) {
            sensor_msgs::msg::PointCloud2 esdf_msg;
            esdf_msg.header.stamp = last_header_.stamp;
            esdf_msg.header.frame_id = target_frame_;
            auto cloud = esdf_->get_occupied_points(5);
            pc_pub_.publish(cloud, esdf_msg, ESDF_MAP_TOPIC);
        }
        if (utils::publisher_sub(grid_map_pub_)) {
            do {
                nav_msgs::msg::OccupancyGrid msg;
                auto& bin_voxel = bin_map_->voxel_map_;
                auto min_key = bin_voxel->min_key;
                auto max_key = bin_voxel->max_key;
                const int width = max_key.x() - min_key.x() + 1;
                const int height = max_key.y() - min_key.y() + 1;

                if (width <= 0 || height <= 0) {
                    break;
                }

                msg.header.stamp = last_header_.stamp;
                msg.header.frame_id = target_frame_;

                msg.info.resolution = bin_voxel->voxel_size;
                msg.info.width = width;
                msg.info.height = height;

                msg.info.origin.position.x = min_key.x() * bin_voxel->voxel_size;

                msg.info.origin.position.y = min_key.y() * bin_voxel->voxel_size;

                msg.info.origin.position.z = 0.0;

                msg.info.origin.orientation.w = 1.0;

                msg.data.assign(width * height, 0); // unknown

                for (const auto& [h, cell]: bin_voxel->grid) {
                    auto key = bin_voxel->hash_to_key(h);
                    int x = key.x() - min_key.x();
                    int y = key.y() - min_key.y();

                    if (x < 0 || x >= width || y < 0 || y >= height)
                        continue; // 安全保护

                    const int idx = y * width + x;
                    msg.data[idx] = 100;
                }
                grid_map_pub_->publish(msg);
            } while (0);
        }
    }

    void bin_callback(BinMap* bin) noexcept {
        if (has_added_static_) {
            return;
        }
        auto& bin_voxel = bin->voxel_map_;
        auto voxel_3d = occ_map_->get_voxel_map();
        static std::unique_ptr<SlidingVoxelMap<2, float>> slope_map;
        constexpr double _diff = -0.5;
        if (!slope_map) {
            auto max_k_3d = voxel_3d->max_key;
            auto max_p_3d = voxel_3d->key_to_world(max_k_3d);
            auto min_k_3d = voxel_3d->min_key;
            auto min_p_3d = voxel_3d->key_to_world(min_k_3d);
            min_p_3d.x() += _diff;
            min_p_3d.y() += _diff;
            max_p_3d.x() -= _diff;
            max_p_3d.y() -= _diff;
            slope_map = std::make_unique<SlidingVoxelMap<2, float>>(
                bin_map_->voxel_map_->voxel_size,
                min_p_3d.head<2>(),
                max_p_3d.head<2>(),
                true
            );
        }

        slope_map->slide_to(
            slope_map->world_to_key(voxel_3d->get_center().head<2>()),
            [&](int idx) {

            }
        );

        auto occupied_idxs_3d = occ_map_->get_occupied_idx();
        if (occupied_idxs_3d.empty()) {
            return;
        }
        auto pointcloud = std::make_shared<small_gicp::PointCloud>();
        pointcloud->resize(occupied_idxs_3d.size());
        for (int i = 0; i < occupied_idxs_3d.size(); i++) {
            auto w3d = voxel_3d->index_to_world(occupied_idxs_3d[i]);
            pointcloud->point(i) << w3d.cast<double>(), 1.0;
        }
        small_gicp::estimate_normals_tbb(*pointcloud, 20);
        double robo_z = voxel_3d->get_center().z();
        double min_z = robo_z + bin_params_.bottom_z_to_robo_z;
        double max_z = robo_z + bin_params_.top_z_to_robo_z;
        std::vector<int> count(slope_map->grid.size(), 0);

        for (size_t i = 0; i < pointcloud->size(); ++i) {
            const auto& pt = pointcloud->point(i);
            const auto& n = pointcloud->normal(i);
            if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z()))
                continue;
            float cos_theta = std::abs(n.z());
            int idx2d = slope_map->world_to_index(Eigen::Vector2f(pt.x(), pt.y()));
            if (idx2d < 0 || pt.z() < min_z || pt.z() > max_z)
                continue;
            slope_map->grid[idx2d] += cos_theta;
            count[idx2d]++;
        }
        for (auto it = bin_voxel->grid.begin(); it != bin_voxel->grid.end();) {
            auto p = bin_voxel->hash_to_world(it->first);
            // if (slope_map->world_to_index(p) >= 0) {
            //     if (!it->second.is_static) {
            //         it = bin_voxel->grid.erase(it);
            //         continue;
            //     }
            // }
            if (!it->second.is_static) {
                it = bin_voxel->grid.erase(it);
                continue;
            }
            ++it;
        }
        for (int i = 0; i < slope_map->grid.size(); ++i) {
            if (count[i] <= bin_params_.count_thresh) {
                slope_map->grid[i] = 0.0;
                continue;
            }
            auto p = slope_map->index_to_world(i);
            auto it = bin_voxel->get_cell(p);
            if (it && it->is_static) {
                slope_map->grid[i] = 0.0;
                continue;
            }
            float avg_cos = slope_map->grid[i] / count[i];
            slope_map->grid[i] = 0.0;
            if (avg_cos < bin_params_.cos_thresh) {
                bin_voxel->set_cell(p, BinMap::Cell { .is_static = false });
            }
        }

        return;
    }
    std::thread process_thread_;
    OccMap::Ptr occ_map_;
    BinMap::Ptr bin_map_;
    ESDF::Ptr esdf_;
    std::string sensor_frame_;
    std::string target_frame_;
    double max_update_dt_ = 0.1;
    std_msgs::msg::Header last_header_;
    rclcpp::Node* node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    PcPub pc_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr add_static_;
    double current_time_ = 0.0;
    RclTF::Ptr tf_;
    bool has_added_static_ = false;
};
RoseMap::RoseMap(rclcpp::Node& node) {
    _impl = std::make_unique<Impl>(node);
}
RoseMap::~RoseMap() {
    _impl.reset();
}
OccMap::Ptr RoseMap::occ_map() const {
    return _impl->occ_map_;
}
BinMap::Ptr RoseMap::bin_map() const {
    return _impl->bin_map_;
}
ESDF::Ptr RoseMap::esdf() const {
    return _impl->esdf_;
}
} // namespace rose_nav::map