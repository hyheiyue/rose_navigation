#include "small_point_lio.hpp"
#include "estimator.h"
#include "lidar_adapter/base_lidar.h"
#include "lidar_adapter/custom_mid360_driver.h"
#include "lidar_adapter/livox_custom_msg.h"
#include "lidar_adapter/livox_pointcloud2.h"
#include "lidar_adapter/unitree_lidar.h"
#include "lidar_adapter/velodyne_pointcloud2.h"
#include "lm/lidar_adapter/base_lidar.h"
#include "param_deliver.h"
#include "utils/io/pcd_io.h"
#include "utils/mapping/pcd_mapping.h"
#include "utils/rcl_tf.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <deque>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

namespace rose_nav::lm {
struct SmallPointLIO::Impl {
    struct Params {
        std::string lidar_frame;
        std::string robot_base_frame;
        std::string base_frame;
        bool save_pcd;
        bool batch_update;
        double min_distance_squared;
        double max_distance_squared;
        double space_downsample_leaf_size = 0.1;
        double batch_interval = 0.01;
        int point_filter_num = 1;
        void load(const ParamsNode& config) {
            lidar_frame = config.declare<std::string>("lidar_frame");
            robot_base_frame = config.declare<std::string>("robot_base_frame");
            base_frame = config.declare<std::string>("base_frame");
            save_pcd = config.declare<bool>("save_pcd");
            batch_update = config.declare<bool>("batch_update");
            auto min_distance = config.declare<double>("min_distance");
            min_distance_squared = min_distance * min_distance;
            auto max_distance = config.declare<double>("max_distance");
            max_distance_squared = max_distance * max_distance;
            space_downsample_leaf_size = config.declare<double>("space_downsample_leaf_size");
            batch_interval = config.declare<double>("batch_interval");
            point_filter_num = config.declare<int>("point_filter_num");
        }
    } params_;
    Impl(rclcpp::Node& node) {
        node_ = &node;
        tf_ = std::make_unique<RclTF>(node);
        auto config = ParamsNode(node, "small_point_lio");
        params_.load(config);
        if (params_.save_pcd) {
            RCLCPP_INFO(rclcpp::get_logger("rose_nav::lm"), "enable save pcd ");
            pcd_mapping = std::make_unique<utils::PCDMapping>(0.05);
        }
        estimator_ = std::make_unique<Estimator>(config);
        Q = estimator_->process_noise_cov();
        std::string imu_topic = config.declare<std::string>("imu_topic");
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic,
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                common::ImuMsg imu_msg;
                imu_msg.angular_velocity = Eigen::Vector3d(
                    msg->angular_velocity.x,
                    msg->angular_velocity.y,
                    msg->angular_velocity.z
                );
                imu_msg.linear_acceleration = Eigen::Vector3d(
                    msg->linear_acceleration.x,
                    msg->linear_acceleration.y,
                    msg->linear_acceleration.z
                );
                imu_msg.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
                on_imu_callback(imu_msg);
                handle_once();
                log_ctx_.imu_cbk_count++;
            }
        );
        std::string lidar_topic = config.declare<std::string>("lidar_topic");
        std::string lidar_type = config.declare<std::string>("lidar_type");
        if (lidar_type == "livox_custom_msg") {
#ifdef HAVE_LIVOX_DRIVER
            lidar_adapter = std::make_unique<LivoxCustomMsgAdapter>();
#else
            RCLCPP_ERROR(
                rclcpp::get_logger("rose_nav::lm"),
                "livox_custom_msg requested but not available!"
            );
            rclcpp::shutdown();
            return;
#endif
        } else if (lidar_type == "livox_pointcloud2") {
            lidar_adapter_ = std::make_unique<LivoxPointCloud2Adapter>();
        } else if (lidar_type == "custom_mid360_driver") {
            lidar_adapter_ = std::make_unique<CustomMid360DriverAdapter>();
        } else if (lidar_type == "unilidar") {
            lidar_adapter_ = std::make_unique<UnilidarAdapter>();
        } else if (lidar_type == "velodyne") {
            lidar_adapter_ = std::make_unique<VelodynePointCloud2>();
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("rose_nav::lm"), "unknwon lidar type");
            rclcpp::shutdown();
            return;
        }
        lidar_adapter_->setup_subscription(
            node_,
            lidar_topic,
            [this](const std::vector<common::Point>& pointcloud, const rclcpp::Time&) {
                on_point_cloud_callback(pointcloud);
                handle_once();
                log_ctx_.pointcloud_cbk_count++;
            }
        );

        map_save_trigger_ = node_->create_service<std_srvs::srv::Trigger>(
            "map_save",
            [this](
                const std_srvs::srv::Trigger::Request::SharedPtr /**/,
                std_srvs::srv::Trigger::Response::SharedPtr /**/
            ) {
                if (!pcd_mapping) {
                    RCLCPP_ERROR(rclcpp::get_logger("rose_nav::lm"), "pcd save is disabled");
                }
                RCLCPP_INFO(rclcpp::get_logger("rose_nav::lm"), "waiting for pcd saving ...");
                auto pointcloud_to_save = std::make_shared<std::vector<Eigen::Vector3f>>();
                *pointcloud_to_save = pcd_mapping->get_points();
                std::thread([pointcloud_to_save]() {
                    io::pcd::write_pcd(ROOT_DIR + "/pcd/scan.pcd", *pointcloud_to_save);
                    RCLCPP_INFO(rclcpp::get_logger("rose_nav::lm"), "save pcd success");
                }).detach();
            }
        );
        marker_pub_ =
            node_->create_publisher<visualization_msgs::msg::MarkerArray>("/lm_marker", 10);
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 1000);
        pointcloud_pub_ =
            node_->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
    }
    struct LogCtx {
        int imu_cbk_count = 0;
        int pointcloud_cbk_count = 0;
        int processed_points = 0;
        double cost_ms = 0;
        void reset() {
            imu_cbk_count = 0;
            pointcloud_cbk_count = 0;
            processed_points = 0;
            cost_ms = 0;
        }
    } log_ctx_;
    void handle_once() {
        static double time_current = -1.0;
        static bool is_inited = false;
        std::vector<Eigen::Vector3f> pointcloud_odom_frame;

        auto start = std::chrono::steady_clock::now();

        auto has_lidar = [&]() -> bool {
            return params_.batch_update ? !preprocess_.point_batch_deque.empty()
                                        : !preprocess_.point_deque.empty();
        };

        auto lidar_timestamp = [&]() -> double {
            return params_.batch_update ? preprocess_.point_batch_deque.front().timestamp
                                        : preprocess_.point_deque.front().timestamp;
        };

        auto has_dense = [&]() -> bool {
            return !use_dense_points() || !preprocess_.dense_point_deque.empty();
        };

        if (!is_inited) {
            int total_points = 0;
            if (params_.batch_update) {
                for (const auto& b: preprocess_.point_batch_deque) {
                    total_points += static_cast<int>(b.points.size());
                }
            } else {
                total_points = static_cast<int>(preprocess_.point_deque.size());
            }

            if ((!preprocess_.imu_deque.empty() || total_points > 0)
                && total_points >= estimator_->params_.init_map_size
                && (!estimator_->params_.fix_gravity_direction
                    || preprocess_.imu_deque.size() >= 200))
            {
                if (params_.batch_update) {
                    for (const auto& batch: preprocess_.point_batch_deque) {
                        for (const auto& point: batch.points) {
                            estimator_->ivox->add_point(point.position);
                        }
                    }
                } else {
                    for (const auto& point: preprocess_.point_deque) {
                        estimator_->ivox->add_point(point.position);
                    }
                }

                if (estimator_->params_.fix_gravity_direction) {
                    estimator_->kf.x.gravity = Eigen::Matrix<state::value_type, 3, 1>::Zero();
                    for (const auto& imu_msg: preprocess_.imu_deque) {
                        estimator_->kf.x.gravity +=
                            imu_msg.linear_acceleration.cast<state::value_type>();
                    }

                    state::value_type scale =
                        -static_cast<state::value_type>(estimator_->params_.gravity.norm())
                        / estimator_->kf.x.gravity.norm();
                    estimator_->kf.x.gravity *= scale;
                } else {
                    estimator_->kf.x.gravity =
                        estimator_->params_.gravity.cast<state::value_type>();
                }

                estimator_->kf.x.acceleration = -estimator_->kf.x.gravity;

                if (params_.batch_update) {
                    if (preprocess_.point_batch_deque.empty()) {
                        time_current = preprocess_.imu_deque.back().timestamp;
                    } else if (preprocess_.imu_deque.empty()) {
                        time_current = preprocess_.point_batch_deque.back().timestamp;
                    } else {
                        time_current = std::max(
                            preprocess_.point_batch_deque.back().timestamp,
                            preprocess_.imu_deque.back().timestamp
                        );
                    }
                } else {
                    if (preprocess_.point_deque.empty()) {
                        time_current = preprocess_.imu_deque.back().timestamp;
                    } else if (preprocess_.imu_deque.empty()) {
                        time_current = preprocess_.point_deque.back().timestamp;
                    } else {
                        time_current = std::max(
                            preprocess_.point_deque.back().timestamp,
                            preprocess_.imu_deque.back().timestamp
                        );
                    }
                }

                estimator_->kf.init_timestamp(time_current);

                preprocess_.dense_point_deque.clear();
                preprocess_.point_batch_deque.clear();
                preprocess_.point_deque.clear();
                preprocess_.imu_deque.clear();

                is_inited = true;
            }
            return;
        }

        bool is_publish_odometry = false;
        if (params_.batch_update) {
            is_publish_odometry = !preprocess_.imu_deque.empty()
                && !preprocess_.point_batch_deque.empty()
                && preprocess_.point_batch_deque.front().timestamp
                    < preprocess_.imu_deque.back().timestamp
                && preprocess_.point_batch_deque.back().timestamp
                    > preprocess_.imu_deque.front().timestamp;
        } else {
            is_publish_odometry = !preprocess_.imu_deque.empty() && !preprocess_.point_deque.empty()
                && preprocess_.imu_deque.front().timestamp
                    < preprocess_.point_deque.back().timestamp;
        }

        const bool use_dense = use_dense_points();

        while (has_lidar() && !preprocess_.imu_deque.empty() && has_dense()) {
            const double imu_ts = preprocess_.imu_deque.front().timestamp;
            const double lidar_ts = lidar_timestamp();
            const double dense_ts = (use_dense && !preprocess_.dense_point_deque.empty())
                ? preprocess_.dense_point_deque.front().timestamp
                : std::numeric_limits<double>::infinity();

            if (use_dense && dense_ts < lidar_ts && dense_ts < imu_ts) {
                const common::Point& dense_point_lidar_frame =
                    preprocess_.dense_point_deque.front();

                Eigen::Matrix<state::value_type, 3, 1> dense_point_imu_frame;
                if (estimator_->params_.extrinsic_est_en) {
                    dense_point_imu_frame = estimator_->kf.x.offset_R_L_I
                            * dense_point_lidar_frame.position.cast<state::value_type>()
                        + estimator_->kf.x.offset_T_L_I;
                } else {
                    dense_point_imu_frame = estimator_->Lidar_R_wrt_IMU
                            * dense_point_lidar_frame.position.cast<state::value_type>()
                        + estimator_->Lidar_T_wrt_IMU;
                }

                pointcloud_odom_frame.emplace_back(
                    (estimator_->kf.x.rotation * dense_point_imu_frame + estimator_->kf.x.position)
                        .cast<float>()
                );
                preprocess_.dense_point_deque.pop_front();
                continue;
            }

            if (lidar_ts < imu_ts) {
                if (params_.batch_update) {
                    const common::Batch& batch_frame = preprocess_.point_batch_deque.front();

                    if (batch_frame.timestamp < time_current) {
                        preprocess_.point_batch_deque.pop_front();
                        continue;
                    }

                    time_current = batch_frame.timestamp;
                    estimator_->kf.predict_state(time_current);
                    estimator_->current_batch = batch_frame;
                    estimator_->kf.update_iterated_batch();

                    for (const auto& point: estimator_->points_odom_frame) {
                        estimator_->ivox->add_point(point);
                    }

                    log_ctx_.processed_points += estimator_->points_odom_frame.size();
                    preprocess_.point_batch_deque.pop_front();
                } else {
                    const common::Point& point_lidar_frame = preprocess_.point_deque.front();

                    if (point_lidar_frame.timestamp < time_current || point_lidar_frame.count < 1) {
                        preprocess_.point_deque.pop_front();
                        continue;
                    }

                    time_current = point_lidar_frame.timestamp;
                    estimator_->kf.predict_state(time_current);
                    estimator_->point_lidar_frame = point_lidar_frame.position;
                    estimator_->kf.update_point();
                    estimator_->ivox->add_point(estimator_->point_odom_frame);

                    log_ctx_.processed_points++;
                    preprocess_.point_deque.pop_front();
                }
                continue;
            } else {
                const common::ImuMsg& imu_msg = preprocess_.imu_deque.front();

                if (imu_msg.timestamp < time_current) {
                    preprocess_.imu_deque.pop_front();
                    continue;
                }

                time_current = imu_msg.timestamp;

                estimator_->kf.predict_state(time_current);
                estimator_->kf.predict_cov(time_current, Q);

                estimator_->angular_velocity = imu_msg.angular_velocity.cast<state::value_type>();
                estimator_->linear_acceleration =
                    imu_msg.linear_acceleration.cast<state::value_type>();
                estimator_->kf.update_imu();

                preprocess_.imu_deque.pop_front();
            }
        }

        if (is_publish_odometry) {
            common::Odometry odometry;
            odometry.timestamp = time_current;
            odometry.position = estimator_->kf.x.position.cast<double>();
            odometry.velocity = estimator_->kf.x.velocity.cast<double>();
            odometry.orientation = estimator_->kf.x.rotation.cast<double>();
            odometry.angular_velocity = estimator_->kf.x.omg.cast<double>();

            publish_odometry(odometry);

            if (!pointcloud_odom_frame.empty()) {
                publish_pointCloud(pointcloud_odom_frame);
            }

            pointcloud_odom_frame.clear();
        }

        auto end = std::chrono::steady_clock::now();
        auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        log_ctx_.cost_ms += cost;

        utils::dt_once(
            [&]() {
                RCLCPP_INFO_STREAM(
                    rclcpp::get_logger("rose_nav::lm"),
                    "imu: " << log_ctx_.imu_cbk_count << " pc: " << log_ctx_.pointcloud_cbk_count
                            << " process: " << log_ctx_.processed_points
                            << " cost: " << log_ctx_.cost_ms << "ms"
                );
                log_ctx_.reset();
            },
            std::chrono::duration<double>(1.0)
        );
    }
    bool is_first_pointcloud_ = true;
    tf2::Transform tf_odom_to_lodom_;
    common::Odometry last_odometry_;
    void publish_pointCloud(const std::vector<Eigen::Vector3f>& pointcloud) {
        if (utils::publisher_sub(pointcloud_pub_)) {
            builtin_interfaces::msg::Time time_msg;
            time_msg.sec = std::floor(last_odometry_.timestamp);
            time_msg.nanosec =
                static_cast<uint32_t>((last_odometry_.timestamp - time_msg.sec) * 1e9);
            geometry_msgs::msg::TransformStamped lidar_frame_to_base_link_transform;
            tf2::Transform tf_in_odom = tf_odom_to_lodom_;
            lidar_frame_to_base_link_transform.transform = tf2::toMsg(tf_in_odom);
            lidar_frame_to_base_link_transform.header.frame_id = "odom";
            Eigen::Vector3f lidar_frame_to_base_link_T;
            lidar_frame_to_base_link_T
                << static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.x),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.y),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.z);
            Eigen::Matrix3f lidar_frame_to_base_link_R =
                Eigen::Quaternionf(
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.w),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.x),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.y),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.z)
                )
                    .toRotationMatrix();
            sensor_msgs::msg::PointCloud2 msg;
            msg.header.stamp = time_msg;
            msg.header.frame_id = "odom";
            msg.width = pointcloud.size();
            msg.height = 1;
            msg.fields.reserve(4);
            sensor_msgs::msg::PointField field;
            field.name = "x";
            field.offset = 0;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "y";
            field.offset = 4;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "z";
            field.offset = 8;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "intensity";
            field.offset = 12;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            msg.is_bigendian = false;
            msg.point_step = 16;
            msg.row_step = msg.width * msg.point_step;
            msg.data.resize(msg.row_step * msg.height);
            Eigen::Vector3f transformed_point;
            auto pointer = reinterpret_cast<float*>(msg.data.data());
            for (const auto& point: pointcloud) {
                transformed_point = lidar_frame_to_base_link_R * point + lidar_frame_to_base_link_T;
                *pointer = transformed_point.x();
                ++pointer;
                *pointer = transformed_point.y();
                ++pointer;
                *pointer = transformed_point.z();
                ++pointer;
                *pointer = (point - last_odometry_.position.cast<float>()).norm();
                ++pointer;
                if (pcd_mapping)
                    pcd_mapping->add_point(transformed_point);
            }
            msg.is_dense = false;
            pointcloud_pub_->publish(msg);
        } else if (pcd_mapping) {
            geometry_msgs::msg::TransformStamped lidar_frame_to_base_link_transform;
            tf2::Transform tf_in_odom = tf_odom_to_lodom_;
            lidar_frame_to_base_link_transform.transform = tf2::toMsg(tf_in_odom);
            Eigen::Vector3f lidar_frame_to_base_link_T;
            lidar_frame_to_base_link_T
                << static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.x),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.y),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.z);
            Eigen::Matrix3f lidar_frame_to_base_link_R =
                Eigen::Quaternionf(
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.w),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.x),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.y),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.z)
                )
                    .toRotationMatrix();
            Eigen::Vector3f transformed_point;
            for (const auto& point: pointcloud) {
                transformed_point = lidar_frame_to_base_link_R * point + lidar_frame_to_base_link_T;
                pcd_mapping->add_point(transformed_point);
            }
        }
    }

    void publish_odometry(const common::Odometry& odometry) {
        last_odometry_ = odometry;
        builtin_interfaces::msg::Time time_msg;
        time_msg.sec = std::floor(odometry.timestamp);
        time_msg.nanosec = static_cast<uint32_t>((odometry.timestamp - time_msg.sec) * 1e9);
        tf2::Transform tf_lidar_to_lodom;
        tf_lidar_to_lodom.setOrigin(
            tf2::Vector3(odometry.position.x(), odometry.position.y(), odometry.position.z())
        );
        tf_lidar_to_lodom.setRotation(tf2::Quaternion(
            odometry.orientation.x(),
            odometry.orientation.y(),
            odometry.orientation.z(),
            odometry.orientation.w()
        ));
        if (is_first_pointcloud_) {
            auto l2r_opt = tf_->get_tf2_transform(
                params_.lidar_frame,
                params_.robot_base_frame,
                time_msg,
                rclcpp::Duration::from_seconds(1.0)
            );
            if (l2r_opt) {
                tf_odom_to_lodom_ = l2r_opt.value().inverse();
                is_first_pointcloud_ = false;
            } else {
                return;
            }
        }
        static tf2::Transform tf_lidar_to_robot;
        static tf2::Transform tf_robot_to_base;
        auto tf_lidar_to_robot_opt = tf_->get_tf2_transform(
            params_.lidar_frame,
            params_.robot_base_frame,
            time_msg,
            rclcpp::Duration::from_seconds(0.1)
        );
        if (tf_lidar_to_robot_opt) {
            tf_lidar_to_robot = tf_lidar_to_robot_opt.value();
        }
        auto tf_robot_to_base_opt = tf_->get_tf2_transform(
            params_.robot_base_frame,
            params_.base_frame,
            time_msg,
            rclcpp::Duration::from_seconds(0.1)
        );
        if (tf_robot_to_base_opt) {
            tf_robot_to_base = tf_robot_to_base_opt.value();
        }
        tf2::Transform tf_robot_to_odom =
            tf_lidar_to_robot.inverse() * tf_lidar_to_lodom * tf_odom_to_lodom_.inverse();
        auto tf_base_to_odom = tf_robot_to_base * tf_robot_to_odom;
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = time_msg;
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id = params_.base_frame;
        tf_msg.transform = tf2::toMsg(tf_base_to_odom);
        tf_->publish_transform(tf_msg);

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = time_msg;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = params_.robot_base_frame;
        const auto& t = tf_robot_to_odom.getOrigin();
        const auto& q = tf_robot_to_odom.getRotation();
        odom_msg.pose.pose.position.x = t.x();
        odom_msg.pose.pose.position.y = t.y();
        odom_msg.pose.pose.position.z = t.z();
        odom_msg.pose.pose.orientation = tf2::toMsg(q);
        static tf2::Transform last_tf;
        static rclcpp::Time last_stamp = rclcpp::Time(time_msg);
        if (last_stamp.nanoseconds() > 0) {
            rclcpp::Time current_time(time_msg);
            double dt = (current_time - last_stamp).seconds();
            if (dt > 1e-6) {
                auto diff = tf_robot_to_odom.getOrigin() - last_tf.getOrigin();
                odom_msg.twist.twist.linear.x = diff.x() / dt;
                odom_msg.twist.twist.linear.y = diff.y() / dt;
                odom_msg.twist.twist.linear.z = diff.z() / dt;

                tf2::Quaternion dq =
                    tf_robot_to_odom.getRotation() * last_tf.getRotation().inverse();
                tf2::Vector3 axis = dq.getAxis();
                double angle = dq.getAngle();
                tf2::Vector3 ang_vel = axis * angle / dt;
                odom_msg.twist.twist.angular.x = ang_vel.x();
                odom_msg.twist.twist.angular.y = ang_vel.y();
                odom_msg.twist.twist.angular.z = ang_vel.z();
            }
        }
        last_tf = tf_robot_to_odom;
        last_stamp = time_msg;

        visualization_msgs::msg::Marker linear_v_marker;
        linear_v_marker.type = visualization_msgs::msg::Marker::ARROW;
        linear_v_marker.ns = "linear_v";
        linear_v_marker.scale.x = 0.1;
        linear_v_marker.scale.y = 0.1;
        linear_v_marker.color.a = 1.0;
        linear_v_marker.color.r = 0.0;
        linear_v_marker.color.g = 1.0;
        linear_v_marker.color.b = 0.0;
        linear_v_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        linear_v_marker.header = odom_msg.header;
        linear_v_marker.action = visualization_msgs::msg::Marker::ADD;
        linear_v_marker.id = 1;
        linear_v_marker.points.clear();
        linear_v_marker.points.emplace_back(odom_msg.pose.pose.position);
        geometry_msgs::msg::Point arrow_end = odom_msg.pose.pose.position;
        arrow_end.x += odom_msg.twist.twist.linear.x;
        arrow_end.y += odom_msg.twist.twist.linear.y;
        arrow_end.z += odom_msg.twist.twist.linear.z;
        linear_v_marker.points.emplace_back(arrow_end);
        visualization_msgs::msg::MarkerArray marker_array;
        marker_array.markers.push_back(linear_v_marker);
        marker_pub_->publish(marker_array);
        odom_pub_->publish(odom_msg);
    }

    void voxelgrid_sampling_tbb(
        const std::vector<common::Point>& points,
        std::vector<common::Point>& downsampled,
        double leaf_size
    ) const noexcept {
        static std::vector<std::pair<std::uint64_t, size_t>> coord_pt;
        if (points.empty()) {
            downsampled = points;
            return;
        }

        const size_t N = points.size();
        const double inv_leaf_size = 1.0 / leaf_size;

        constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
        constexpr int coord_bit_size = 21;
        constexpr std::uint64_t coord_bit_mask = (1ull << coord_bit_size) - 1;
        constexpr int coord_offset = 1 << (coord_bit_size - 1);

        coord_pt.resize(N);
        auto fast_floor = [](const Eigen::Array3f& pt) -> Eigen::Array3i {
            const Eigen::Array3i ncoord = pt.cast<int>();
            return ncoord - (pt < ncoord.cast<float>()).cast<int>();
        };
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, N, 2048),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& pt = points[i];

                    Eigen::Array3i coord = fast_floor(pt.position * inv_leaf_size) + coord_offset;

                    if ((coord < 0).any() || (coord > static_cast<int>(coord_bit_mask)).any()) {
                        coord_pt[i] = { invalid_coord, i };
                        continue;
                    }

                    std::uint64_t bits =
                        (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << 0)
                        | (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << 21)
                        | (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << 42);

                    coord_pt[i] = { bits, i };
                }
            }
        );

        tbb::parallel_sort(coord_pt.begin(), coord_pt.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        std::vector<size_t> voxel_starts;
        voxel_starts.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            if (coord_pt[i].first == invalid_coord)
                continue;

            if (i == 0 || coord_pt[i].first != coord_pt[i - 1].first) {
                voxel_starts.push_back(i);
            }
        }

        if (voxel_starts.empty()) {
            downsampled.clear();
            return;
        }

        downsampled.resize(voxel_starts.size());

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, voxel_starts.size(), 2048),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t vi = r.begin(); vi != r.end(); ++vi) {
                    size_t start = voxel_starts[vi];
                    size_t end = (vi + 1 < voxel_starts.size()) ? voxel_starts[vi + 1] : N;
                    const auto& first_pt = points[coord_pt[start].second];

                    Eigen::Vector3f sum_pos = first_pt.position;

                    double timestamp = first_pt.timestamp;

                    int count = 1;
                    for (size_t i = start + 1; i < end; ++i) {
                        if (coord_pt[i].first == invalid_coord)
                            continue;

                        const auto& pt = points[coord_pt[i].second];

                        sum_pos += pt.position;
                        timestamp = pt.timestamp;
                        ++count;
                    }

                    common::Point out = first_pt;

                    out.position = sum_pos / static_cast<double>(count);

                    out.timestamp = timestamp;
                    out.count = count;

                    downsampled[vi] = std::move(out);
                }
            }
        );
    }
    bool use_dense_points() {
        return utils::publisher_sub(pointcloud_pub_) || pcd_mapping;
    }
    void on_point_cloud_callback(const std::vector<common::Point>& pointcloud) {
        static double last_batch_timestamp = -1.0;
        static double last_lidar_timestamp = -1.0;
        static double last_timestamp_dense_point = -1.0;
        static std::vector<common::Point> filtered_points;
        static std::vector<common::Point> processed_pointcloud;
        static std::vector<common::Point> dense_points;
        static common::Batch current_batch;
        current_batch.points.clear();
        filtered_points.clear();
        filtered_points.reserve(pointcloud.size());
        dense_points.clear();
        dense_points.reserve(pointcloud.size());
        double space_downsample_leaf_size = params_.space_downsample_leaf_size;
        for (size_t i = 0; i < pointcloud.size(); i++) {
            const auto& point = pointcloud[i];
            float dist = point.position.squaredNorm();
            if (dist < params_.min_distance_squared || dist > params_.max_distance_squared) {
                continue;
            }
            if (point.timestamp >= last_timestamp_dense_point && use_dense_points()) {
                dense_points.push_back(point);
            }
            if (i % params_.point_filter_num != 0) {
                continue;
            }
            if (point.timestamp < last_lidar_timestamp) {
                continue;
            }
            filtered_points.push_back(point);
        }
        if (space_downsample_leaf_size >= 0.01) {
            voxelgrid_sampling_tbb(
                filtered_points,
                processed_pointcloud,
                space_downsample_leaf_size
            );
        } else {
            processed_pointcloud = std::move(filtered_points);
        }
        tbb::parallel_sort(
            dense_points.begin(),
            dense_points.end(),
            [](const auto& x, const auto& y) { return x.timestamp < y.timestamp; }
        );
        tbb::parallel_sort(
            processed_pointcloud.begin(),
            processed_pointcloud.end(),
            [](const auto& x, const auto& y) { return x.timestamp < y.timestamp; }
        );
        if (!dense_points.empty()) {
            last_timestamp_dense_point = dense_points.back().timestamp;
            preprocess_.dense_point_deque.insert(
                preprocess_.dense_point_deque.end(),
                dense_points.begin(),
                dense_points.end()
            );
        }
        auto mean_timestamp = [](const std::vector<common::Point>& points) {
            double sum = std::accumulate(
                points.begin(),
                points.end(),
                0.0,
                [](double acc, const common::Point& p) { return acc + p.timestamp; }
            );
            return sum / static_cast<double>(points.size());
        };
        if (!processed_pointcloud.empty()) {
            last_lidar_timestamp = processed_pointcloud.back().timestamp;
            if (params_.batch_update) {
                for (const auto& p: processed_pointcloud) {
                    if (current_batch.points.empty()) {
                        current_batch.points.push_back(p);
                        last_batch_timestamp = p.timestamp;
                    } else if (p.timestamp - last_batch_timestamp < params_.batch_interval) {
                        current_batch.points.push_back(p);
                    } else {
                        current_batch.timestamp = mean_timestamp(current_batch.points);
                        preprocess_.point_batch_deque.push_back(current_batch);
                        current_batch = common::Batch();
                        current_batch.points.push_back(p);
                        last_batch_timestamp = p.timestamp;
                    }
                }
                if (!current_batch.points.empty()) {
                    current_batch.timestamp = mean_timestamp(current_batch.points);
                    preprocess_.point_batch_deque.push_back(current_batch);
                    current_batch = common::Batch();
                }
            } else {
                preprocess_.point_deque.insert(
                    preprocess_.point_deque.end(),
                    processed_pointcloud.begin(),
                    processed_pointcloud.end()
                );
            }
        }
    }
    void on_imu_callback(const common::ImuMsg& imu_msg) {
        static double last_imu_timestamp = -1.0;
        if (imu_msg.timestamp < last_imu_timestamp) {
            RCLCPP_ERROR(rclcpp::get_logger("rose_nav:lm"), "imu loop back");
            return;
        }
        preprocess_.imu_deque.emplace_back(imu_msg);
        last_imu_timestamp = imu_msg.timestamp;
    }
    struct Preprocess {
        std::deque<common::Point> point_deque;
        std::deque<common::ImuMsg> imu_deque;
        std::deque<common::Batch> point_batch_deque;
        std::deque<common::Point> dense_point_deque;
    } preprocess_;

    rclcpp::Node* node_;
    Estimator::Ptr estimator_;
    std::unique_ptr<LidarAdapterBase> lidar_adapter_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_trigger_;
    Eigen::Matrix<state::value_type, state::DIM, state::DIM> Q;
    RclTF::Ptr tf_;
    std::unique_ptr<utils::PCDMapping> pcd_mapping;
};
SmallPointLIO::SmallPointLIO(rclcpp::Node& node) {
    _impl = std::make_unique<Impl>(node);
}
SmallPointLIO::~SmallPointLIO() {
    _impl.reset();
}
} // namespace rose_nav::lm