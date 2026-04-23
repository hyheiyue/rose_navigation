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
#include <Eigen/src/Geometry/Transform.h>
#include <chrono>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tf2/LinearMath/Transform.hpp>
#include <thread>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>
//ros2 service call /map_save std_srvs/srv/Trigger
namespace rose_nav::lm {
struct SmallPointLIO::Impl {
    struct Params {
        std::string lidar_frame;
        std::string robot_base_frame;
        std::string base_frame;
        bool robo_to_lidar_dynamic = false;
        bool save_pcd;
        bool batch_update;
        double min_distance_squared;
        double max_distance_squared;
        double space_downsample_leaf_size = 0.1;
        double batch_interval = 0.01;
        int point_filter_num = 1;
        bool use_priori_pcd_for_algin = false;
        std::string prior_pcd_path;
        Eigen::Isometry3d init_pose_in_prior_pcd;
        int algin_max_iter;
        double algin_fps;
        double algin_max_sq;
        double algin_leaf_size;
        void load(const ParamsNode& config) {
            lidar_frame = config.declare<std::string>("lidar_frame");
            robot_base_frame = config.declare<std::string>("robot_base_frame");
            base_frame = config.declare<std::string>("base_frame");
            robo_to_lidar_dynamic = config.declare<bool>("robo_to_lidar_dynamic");
            save_pcd = config.declare<bool>("save_pcd");
            batch_update = config.declare<bool>("batch_update");
            auto min_distance = config.declare<double>("min_distance");
            min_distance_squared = min_distance * min_distance;
            auto max_distance = config.declare<double>("max_distance");
            max_distance_squared = max_distance * max_distance;
            space_downsample_leaf_size = config.declare<double>("space_downsample_leaf_size");
            batch_interval = config.declare<double>("batch_interval");
            point_filter_num = config.declare<int>("point_filter_num");
            use_priori_pcd_for_algin = config.declare<bool>("use_priori_pcd_for_algin");
            prior_pcd_path = config.declare<std::string>("prior_pcd_path");
            auto init_pose_in_prior_pcd_config = config.sub("init_pose_in_prior_pcd");
            init_pose_in_prior_pcd = Eigen::Isometry3d::Identity();
            auto init_pose_in_prior_pcd_t_vec =
                init_pose_in_prior_pcd_config.declare<std::vector<double>>("translation");
            init_pose_in_prior_pcd.translation() = Eigen::Vector3d(
                init_pose_in_prior_pcd_t_vec[0],
                init_pose_in_prior_pcd_t_vec[1],
                init_pose_in_prior_pcd_t_vec[2]
            );
            auto init_pose_in_prior_pcd_r_vec =
                init_pose_in_prior_pcd_config.declare<std::vector<double>>("rotation");

            init_pose_in_prior_pcd.linear() << init_pose_in_prior_pcd_r_vec[0],
                init_pose_in_prior_pcd_r_vec[1], init_pose_in_prior_pcd_r_vec[2],
                init_pose_in_prior_pcd_r_vec[3], init_pose_in_prior_pcd_r_vec[4],
                init_pose_in_prior_pcd_r_vec[5], init_pose_in_prior_pcd_r_vec[6],
                init_pose_in_prior_pcd_r_vec[7], init_pose_in_prior_pcd_r_vec[8];
            algin_fps = config.declare<double>("algin_fps");
            algin_max_sq = config.declare<double>("algin_max_sq");
            algin_leaf_size = config.declare<double>("algin_leaf_size");
            algin_max_iter = config.declare<int>("algin_max_iter");
        }
    } params_;
    static constexpr int algin_near_num = 20;
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
                if (!has_lidar_to_robot_transform_) {
                    return;
                }
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
                if (!params_.robo_to_lidar_dynamic) {
                    imu_msg.angular_velocity =
                        lidar_to_robot_init_.linear() * imu_msg.angular_velocity;
                    imu_msg.linear_acceleration =
                        lidar_to_robot_init_.linear() * imu_msg.linear_acceleration;
                }
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
            [this](std::vector<common::Point>& pointcloud, const rclcpp::Time& time_msg) {
                if (!has_lidar_to_robot_transform_) {
                    auto l2r_opt = tf_->get_transform<double>(
                        params_.robot_base_frame,
                        params_.lidar_frame,
                        time_msg,
                        rclcpp::Duration::from_seconds(1.0)
                    );
                    if (l2r_opt) {
                        lidar_to_robot_init_ = l2r_opt.value();
                        auto lidar_odom_to_odom_msg = RclTF::eigen2tf(lidar_to_robot_init_);
                        tf2::fromMsg(lidar_odom_to_odom_msg, lidar_odom_to_odom_tf2_);
                        has_lidar_to_robot_transform_ = true;

                    } else {
                        return;
                    }
                }
                if (!params_.robo_to_lidar_dynamic) {
                    auto lidar_to_robot_init_f = lidar_to_robot_init_.cast<float>();
                    for (auto& p: pointcloud) {
                        p.position = lidar_to_robot_init_f * p.position;
                    }
                }

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
        if (params_.use_priori_pcd_for_algin) {
            std::vector<Eigen::Vector3f> pointcloud;
            if (io::pcd::read_pcd(params_.prior_pcd_path, pointcloud)) {
                RCLCPP_INFO(
                    rclcpp::get_logger("rose_nav::lm"),
                    "pcd: %s loaded",
                    params_.prior_pcd_path.c_str()
                );
                target_cloud_ = std::make_shared<small_gicp::PointCloud>();
                target_cloud_->resize(pointcloud.size());
                for (size_t i = 0; i < pointcloud.size(); ++i) {
                    target_cloud_->point(i) << pointcloud[i].cast<double>(), 1.0;
                }
                small_gicp::voxelgrid_sampling_tbb(*target_cloud_, params_.algin_leaf_size);
                small_gicp::estimate_normals_tbb(*target_cloud_, algin_near_num);
                small_gicp::estimate_covariances_tbb(*target_cloud_, algin_near_num);
                target_tree_ = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
                    target_cloud_,
                    small_gicp::KdTreeBuilderTBB()
                );
                now_pose_in_prior_pcd_ = params_.init_pose_in_prior_pcd;
                register_ = std::make_shared<small_gicp::Registration<
                    small_gicp::GICPFactor,
                    small_gicp::ParallelReductionTBB>>();
                register_->rejector.max_dist_sq = params_.algin_max_sq;
                register_->optimizer.max_iterations = params_.algin_max_iter;
                algin_source_grid_ = std::make_unique<utils::PCDMapping>(params_.algin_leaf_size);
                map_to_odom_pub_thread_ = std::thread([&]() {
                    auto next_tp = std::chrono::steady_clock::now();
                    while (rclcpp::ok()) {
                        next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(1.0 / 250.0)
                        );
                        geometry_msgs::msg::TransformStamped tf_msg;
                        tf_msg.header.stamp = node_->now();
                        tf_msg.header.frame_id = "map";
                        tf_msg.child_frame_id = "odom";
                        auto map_to_odom_msg = RclTF::eigen2tf(now_pose_in_prior_pcd_.inverse());
                        tf_msg.transform = map_to_odom_msg;
                        tf_->publish_transform(tf_msg);
                        std::this_thread::sleep_until(next_tp);
                    }
                });
                algin_thread_ = std::thread([&]() {
                    auto next_tp = std::chrono::steady_clock::now();
                    while (rclcpp::ok()) {
                        next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(1.0 / params_.algin_fps)
                        );
                        algin_callback();
                        std::this_thread::sleep_until(next_tp);
                    }
                });

            } else {
                RCLCPP_ERROR(
                    rclcpp::get_logger("rose_nav::lm"),
                    "Failed to load pcd: %s",
                    params_.prior_pcd_path.c_str()
                );
            }
        }
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
        static std::vector<Eigen::Vector3f> pointcloud_odom_frame;

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
                if (params_.batch_update && !estimator_->params_.use_priori_pcd_add_ivox) {
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
                    // if (!estimator_->params_.use_priori_pcd) {
                    for (const auto& point: estimator_->points_odom_frame) {
                        estimator_->ivox->add_point(point);
                    }
                    // }

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
                    bool has_matched = estimator_->kf.update_point();
                    // if (!estimator_->params_.use_priori_pcd) {
                    estimator_->ivox->add_point(estimator_->point_odom_frame);
                    // }

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

    common::Odometry last_odometry_;
    void publish_pointCloud(const std::vector<Eigen::Vector3f>& pointcloud) {
        if (utils::publisher_sub(pointcloud_pub_)) {
            builtin_interfaces::msg::Time time_msg;
            time_msg.sec = std::floor(last_odometry_.timestamp);
            time_msg.nanosec =
                static_cast<uint32_t>((last_odometry_.timestamp - time_msg.sec) * 1e9);
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
                transformed_point = params_.robo_to_lidar_dynamic
                    ? (lidar_to_robot_init_.cast<float>() * point)
                    : point;
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
                if (algin_source_grid_) {
                    std::unique_lock<std::mutex> lock(source_mutex_);
                    algin_source_grid_->add_point(transformed_point);
                }
            }
            msg.is_dense = false;
            pointcloud_pub_->publish(msg);
        } else if (pcd_mapping || algin_source_grid_) {
            for (const auto& point: pointcloud) {
                if (pcd_mapping) {
                    pcd_mapping->add_point(point);
                }
                if (algin_source_grid_) {
                    std::unique_lock<std::mutex> lock(source_mutex_);
                    algin_source_grid_->add_point(point);
                }
            }
        }
    }

    void publish_odometry(const common::Odometry& odometry) {
        last_odometry_ = odometry;
        builtin_interfaces::msg::Time time_msg;
        time_msg.sec = std::floor(odometry.timestamp);
        time_msg.nanosec = static_cast<uint32_t>((odometry.timestamp - time_msg.sec) * 1e9);
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = time_msg;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = params_.robot_base_frame;
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = time_msg;
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id = params_.base_frame;
        if (params_.robo_to_lidar_dynamic) {
            Eigen::Isometry3d lidar_to_lidar_odom = Eigen::Isometry3d::Identity();
            lidar_to_lidar_odom.translation() = odometry.position;
            lidar_to_lidar_odom.linear() = odometry.orientation.toRotationMatrix();
            static tf2::Transform lidar_to_robot_now;
            static tf2::Transform robot_to_base;
            auto lidar_to_robot_now_opt = tf_->get_tf2_transform(
                params_.lidar_frame,
                params_.robot_base_frame,
                time_msg,
                rclcpp::Duration::from_seconds(0.1)
            );
            if (lidar_to_robot_now_opt) {
                lidar_to_robot_now = lidar_to_robot_now_opt.value();
            }
            auto robot_to_base_opt = tf_->get_tf2_transform(
                params_.robot_base_frame,
                params_.base_frame,
                time_msg,
                rclcpp::Duration::from_seconds(0.1)
            );
            if (robot_to_base_opt) {
                robot_to_base = robot_to_base_opt.value();
            }
            auto lidar_to_lidar_odom_msg = RclTF::eigen2tf(lidar_to_lidar_odom);
            tf2::Transform lidar_to_lidar_odom_tf2;
            tf2::fromMsg(lidar_to_lidar_odom_msg, lidar_to_lidar_odom_tf2);
            tf2::Transform tf_robot_to_odom = lidar_to_robot_now.inverse() * lidar_to_lidar_odom_tf2
                * lidar_odom_to_odom_tf2_.inverse();
            auto tf_base_to_odom = robot_to_base * tf_robot_to_odom;

            tf_msg.transform = tf2::toMsg(tf_base_to_odom);

            const auto& t = tf_robot_to_odom.getOrigin();
            const auto& q = tf_robot_to_odom.getRotation();
            odom_msg.pose.pose.position.x = t.x();
            odom_msg.pose.pose.position.y = t.y();
            odom_msg.pose.pose.position.z = t.z();
            odom_msg.pose.pose.orientation = tf2::toMsg(q);
            static tf2::Transform last_tf_robot_to_odom;
            static rclcpp::Time last_stamp = rclcpp::Time(time_msg);
            if (last_stamp.nanoseconds() > 0) {
                rclcpp::Time current_time(time_msg);
                double dt = (current_time - last_stamp).seconds();
                if (dt > 1e-6) {
                    auto diff = tf_robot_to_odom.getOrigin() - last_tf_robot_to_odom.getOrigin();
                    odom_msg.twist.twist.linear.x = diff.x() / dt;
                    odom_msg.twist.twist.linear.y = diff.y() / dt;
                    odom_msg.twist.twist.linear.z = diff.z() / dt;

                    tf2::Quaternion dq = tf_robot_to_odom.getRotation()
                        * last_tf_robot_to_odom.getRotation().inverse();
                    tf2::Vector3 axis = dq.getAxis();
                    double angle = dq.getAngle();
                    tf2::Vector3 ang_vel = axis * angle / dt;
                    odom_msg.twist.twist.angular.x = ang_vel.x();
                    odom_msg.twist.twist.angular.y = ang_vel.y();
                    odom_msg.twist.twist.angular.z = ang_vel.z();
                }
            }
            last_tf_robot_to_odom = tf_robot_to_odom;
            last_stamp = time_msg;
        } else {
            Eigen::Isometry3d robot_to_odom = Eigen::Isometry3d::Identity();
            robot_to_odom.translation() = odometry.position;
            robot_to_odom.linear() = odometry.orientation.toRotationMatrix();
            static tf2::Transform robot_to_base;
            auto robot_to_base_opt = tf_->get_tf2_transform(
                params_.robot_base_frame,
                params_.base_frame,
                time_msg,
                rclcpp::Duration::from_seconds(0.1)
            );
            if (robot_to_base_opt) {
                robot_to_base = robot_to_base_opt.value();
            }
            tf2::Transform tf_robot_to_odom;
            auto robot_to_odom_msg = RclTF::eigen2tf(robot_to_odom);
            tf2::fromMsg(robot_to_odom_msg, tf_robot_to_odom);
            auto tf_base_to_odom = robot_to_base * tf_robot_to_odom;

            tf_msg.transform = tf2::toMsg(tf_base_to_odom);

            odom_msg.pose.pose.position.x = odometry.position.x();
            odom_msg.pose.pose.position.y = odometry.position.y();
            odom_msg.pose.pose.position.z = odometry.position.z();
            odom_msg.pose.pose.orientation.x = odometry.orientation.x();
            odom_msg.pose.pose.orientation.y = odometry.orientation.y();
            odom_msg.pose.pose.orientation.z = odometry.orientation.z();
            odom_msg.pose.pose.orientation.w = odometry.orientation.w();
            odom_msg.twist.twist.linear.x = odometry.velocity.x();
            odom_msg.twist.twist.linear.y = odometry.velocity.y();
            odom_msg.twist.twist.linear.z = odometry.velocity.z();
            odom_msg.twist.twist.angular.x = odometry.angular_velocity.x();
            odom_msg.twist.twist.angular.y = odometry.angular_velocity.y();
            odom_msg.twist.twist.angular.z = odometry.angular_velocity.z();
        }

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
        tf_->publish_transform(tf_msg);
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
    void algin_callback() {
        utils::PCDMapping algin_source_grid;
        {
            std::unique_lock<std::mutex> lock(source_mutex_);
            algin_source_grid = *algin_source_grid_;
        }
        auto source = algin_source_grid.get_points();
        auto source_cloud = std::make_shared<small_gicp::PointCloud>();
        source_cloud->resize(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            source_cloud->point(i) << source[i].cast<double>(), 1.0;
        }
        small_gicp::estimate_normals_tbb(*source_cloud, algin_near_num);
        small_gicp::estimate_covariances_tbb(*source_cloud, algin_near_num);

        auto result = register_->align(
            *target_cloud_,
            *source_cloud,
            *target_tree_,
            now_pose_in_prior_pcd_.inverse()
        );
        if (!result.converged) {
            RCLCPP_ERROR_STREAM(
                rclcpp::get_logger("rose_nav:lm"),
                "GICP did not converge, iter_num: " << result.iterations
            );
        }
        now_pose_in_prior_pcd_ = result.T_target_source.inverse();
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
    Eigen::Isometry3d lidar_to_robot_init_;
    tf2::Transform lidar_odom_to_odom_tf2_;
    bool has_lidar_to_robot_transform_ = false;
    mutable std::mutex source_mutex_;
    std::thread algin_thread_;
    std::thread map_to_odom_pub_thread_;
    std::unique_ptr<utils::PCDMapping> algin_source_grid_;
    std::shared_ptr<
        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionTBB>>
        register_;
    small_gicp::PointCloud::Ptr target_cloud_;
    small_gicp::KdTree<small_gicp::PointCloud>::Ptr target_tree_;
    small_gicp::PointCloud::Ptr source_cloud_;

    Eigen::Isometry3d now_pose_in_prior_pcd_;
};
SmallPointLIO::SmallPointLIO(rclcpp::Node& node) {
    _impl = std::make_unique<Impl>(node);
}
SmallPointLIO::~SmallPointLIO() {
    _impl.reset();
}
} // namespace rose_nav::lm