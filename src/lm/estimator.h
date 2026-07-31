/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#include "eskf.h"
#include "lm/common.hpp"
#include "lm/small_oct_vox.hpp"
#include "small_ivox.h"
#include "utils/rclcpp_parameter_node.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tbb/concurrent_hash_map.h>
#include <vector>
namespace rose_nav::lm {

class Estimator {
public:
    using Ptr = std::unique_ptr<Estimator>;
    struct Params {
        int max_iter = 3;
        double map_resolution = 0.1;
        bool extrinsic_est_en = false;
        int h_batch_threads = 1;
        double laser_point_cov = 0.01;
        double imu_meas_acc_cov = 0.01;
        double imu_meas_omg_cov = 0.01;

        double velocity_cov = 20.0;
        double acceleration_cov = 500.0;
        double omg_cov = 1000.0;
        double ba_cov = 0.0001;
        double bg_cov = 0.0001;
        double plane_threshold = 0.1;
        double match_sqaured = 81.0;
        bool check_satu = true;
        double satu_acc = 3.0;
        double satu_gyro = 35.0;
        double acc_norm = 1.0;
        Eigen::Vector3d extrinsic_T;
        Eigen::Matrix3d extrinsic_R;
        Eigen::Vector3d gravity;
        int init_map_size = 10;
        bool fix_gravity_direction = true;
        void load(const ParamsNode& config) {
            max_iter = config.declare<int>("max_iter");
            map_resolution = config.declare<double>("map_resolution");
            extrinsic_est_en = config.declare<bool>("extrinsic_est_en");
            h_batch_threads = config.declare<int>("h_batch_threads", h_batch_threads);
            laser_point_cov = config.declare<double>("laser_point_cov");
            imu_meas_acc_cov = config.declare<double>("imu_meas_acc_cov");
            imu_meas_omg_cov = config.declare<double>("imu_meas_omg_cov");
            velocity_cov = config.declare<double>("velocity_cov");
            acceleration_cov = config.declare<double>("acceleration_cov");
            omg_cov = config.declare<double>("omg_cov");
            ba_cov = config.declare<double>("ba_cov");
            bg_cov = config.declare<double>("bg_cov");
            plane_threshold = config.declare<double>("plane_threshold");
            match_sqaured = config.declare<double>("match_sqaured");
            check_satu = config.declare<bool>("check_satu");
            satu_acc = config.declare<double>("satu_acc");
            satu_gyro = config.declare<double>("satu_gyro");
            acc_norm = config.declare<double>("acc_norm");
            init_map_size = config.declare<int>("init_map_size");
            fix_gravity_direction = config.declare<bool>("fix_gravity_direction");
            auto extrinsic_T_vec = config.declare<std::vector<double>>("extrinsic_T");
            auto extrinsic_R_vec = config.declare<std::vector<double>>("extrinsic_R");
            extrinsic_T =
                Eigen::Vector3d(extrinsic_T_vec[0], extrinsic_T_vec[1], extrinsic_T_vec[2]);
            extrinsic_R << extrinsic_R_vec[0], extrinsic_R_vec[1], extrinsic_R_vec[2],
                extrinsic_R_vec[3], extrinsic_R_vec[4], extrinsic_R_vec[5], extrinsic_R_vec[6],
                extrinsic_R_vec[7], extrinsic_R_vec[8];
            auto gravity_vec = config.declare<std::vector<double>>("gravity");
            gravity = Eigen::Vector3d(gravity_vec[0], gravity_vec[1], gravity_vec[2]);
        }
    } params_;
    Estimator(const ParamsNode& config);

    eskf kf;
    // h_point / h_batch 使用的局部地图与点到平面匹配缓存。
    std::shared_ptr<SmallOctVox> ivox;
    Eigen::Matrix<state::value_type, 3, 1> Lidar_T_wrt_IMU;
    Eigen::Matrix<state::value_type, 3, 3> Lidar_R_wrt_IMU;
    Eigen::Vector3f point_lidar_frame;
    Eigen::Vector3f point_odom_frame;
    std::vector<Eigen::Vector3f> nearest_points;
    common::Batch current_batch;
    std::vector<Eigen::Vector3f> points_odom_frame;
    struct PlaneInfo {
        Eigen::Vector3d normal;
        double plane_d = 0.0;
        bool valid = false;
    };
    struct VoxelIndexHashCompare {
        [[nodiscard]] static size_t hash(const SmallOctVox::PositionIndex& index) noexcept {
            size_t seed = 0;
            for (int k = 0; k < 3; ++k) {
                const size_t value = std::hash<int> {}(index[k]);
                seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            }
            return seed;
        }

        [[nodiscard]] static bool equal(
            const SmallOctVox::PositionIndex& lhs,
            const SmallOctVox::PositionIndex& rhs
        ) noexcept {
            return (lhs.array() == rhs.array()).all();
        }
    };
    using PlaneCache =
        tbb::concurrent_hash_map<SmallOctVox::PositionIndex, PlaneInfo, VoxelIndexHashCompare>;
    PlaneCache batch_plane_cache_;
    PlaneCache point_plane_cache_;
    static void cache_plane(
        PlaneCache& cache,
        const SmallOctVox::PositionIndex& voxel_idx,
        const Eigen::Vector3d& normal,
        double plane_d,
        bool valid
    );

    // h_imu 使用的当前 IMU 量测缓存，进入滤波器前会做尺度和饱和检查。
    Eigen::Matrix<state::value_type, 3, 1> angular_velocity;
    Eigen::Matrix<state::value_type, 3, 1> linear_acceleration;
    double imu_acceleration_scale;
    bool is_inited = false;
    void reset();

    [[nodiscard]] Eigen::Matrix<state::value_type, state::DIM, state::DIM>
    process_noise_cov() const;

    void h_point(const state& s, point_measurement_result& measurement_result);

    void h_imu(const state& s, imu_measurement_result& measurement_result);
    void h_batch(const state& s, batch_measurement_result& result) noexcept;
};

} // namespace rose_nav::lm
