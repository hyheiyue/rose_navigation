/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "estimator.h"
#include "lm/eskf.h"
#include "lm/small_ivox.h"
#include "lm/small_oct_vox.hpp"
#include "utils/io/pcd_io.h"
#include <tbb/tbb.h>
namespace rose_nav::lm {
constexpr int NUM_MATCH_POINTS = 1000;
constexpr int MIN_MATCH_POINTS = 5;
Estimator::Estimator(const ParamsNode& config) {
    params_.load(config);
    Lidar_R_wrt_IMU = params_.extrinsic_R.cast<state::value_type>();
    Lidar_T_wrt_IMU = params_.extrinsic_T.cast<state::value_type>();
    if (params_.extrinsic_est_en) {
        kf.x.offset_T_L_I = params_.extrinsic_T.cast<state::value_type>();
        kf.x.offset_R_L_I = params_.extrinsic_R.cast<state::value_type>();
    }
    imu_acceleration_scale = params_.gravity.norm() / params_.acc_norm;
    kf.init(
        [this](auto&& s, auto&& measurement_result) { return h_point(s, measurement_result); },
        [this](auto&& s, auto&& measurement_result) { return h_imu(s, measurement_result); },
        [this](auto&& s, auto&& measurement_result) { return h_batch(s, measurement_result); }
    );
    kf.max_iter = params_.max_iter;
    reset();
}

void Estimator::reset() {
    ivox = std::make_shared<SmallOctVox>(params_.map_resolution, 1000000);
    batch_iter_cache_.clear();
    if (params_.use_priori_pcd_add_ivox) {
        std::vector<Eigen::Vector3f> pointcloud;
        if (io::pcd::read_pcd(params_.prior_pcd_path, pointcloud)) {
            RCLCPP_INFO(
                rclcpp::get_logger("rose_nav::lm"),
                "pcd: %s loaded",
                params_.prior_pcd_path.c_str()
            );
            tbb::parallel_sort(
                pointcloud.begin(),
                pointcloud.end(),
                [&](const auto& a, const auto& b) {
                    // 先插入离初始位姿更远的点，减少局部区域过密时对 iVox 桶的早期占用。
                    return (a - params_.init_pose_in_prior_pcd.translation().cast<float>())
                               .squaredNorm()
                        > (b - params_.init_pose_in_prior_pcd.translation().cast<float>())
                              .squaredNorm();
                }
            );
            for (const auto& p: pointcloud) {
                ivox->add_point(p);
            }
            RCLCPP_INFO(
                rclcpp::get_logger("rose_nav::lm"),
                "ivox add %d points",
                static_cast<int>(pointcloud.size())
            );
            kf.x.position = params_.init_pose_in_prior_pcd.translation().cast<state::value_type>();
            kf.x.rotation = params_.init_pose_in_prior_pcd.linear().cast<state::value_type>();
        } else {
            RCLCPP_ERROR(
                rclcpp::get_logger("rose_nav::lm"),
                "Failed to load pcd: %s",
                params_.prior_pcd_path.c_str()
            );
        }
    }
    kf.x.reset();
    // 初始协方差保守设置：姿态/位置较小，IMU bias 与重力方向允许后续量测继续修正。
    kf.P = Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Identity() * 0.01;
    kf.P.block<3, 3>(state::gravity_index, state::gravity_index).diagonal().fill(0.0001);
    kf.P.block<3, 3>(state::bg_index, state::bg_index).diagonal().fill(0.001);
    kf.P.block<3, 3>(state::ba_index, state::ba_index).diagonal().fill(0.001);
    is_inited = false;
}

[[nodiscard]] Eigen::Matrix<state::value_type, state::DIM, state::DIM>
Estimator::process_noise_cov() const {
    Eigen::Matrix<state::value_type, state::DIM, state::DIM> cov =
        Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Zero();
    // 过程噪声按状态块配置，便于独立调节速度、角速度、加速度和 IMU bias 的可信度。
    cov.block<3, 3>(state::velocity_index, state::velocity_index)
        .diagonal()
        .fill(static_cast<state::value_type>(params_.velocity_cov));
    cov.block<3, 3>(state::omg_index, state::omg_index)
        .diagonal()
        .fill(static_cast<state::value_type>(params_.omg_cov));
    cov.block<3, 3>(state::acceleration_index, state::acceleration_index)
        .diagonal()
        .fill(static_cast<state::value_type>(params_.acceleration_cov));
    cov.block<3, 3>(state::bg_index, state::bg_index)
        .diagonal()
        .fill(static_cast<state::value_type>(params_.bg_cov));
    cov.block<3, 3>(state::ba_index, state::ba_index)
        .diagonal()
        .fill(static_cast<state::value_type>(params_.ba_cov));
    return cov;
}

void Estimator::h_batch(const state& s, batch_measurement_result& result) noexcept {
    const size_t N = current_batch.points.size();

    result.reset();

    points_odom_frame.assign(N, Eigen::Vector3f::Zero());
    if (s.batch_iter == 0 || batch_iter_cache_.size() != N) {
        batch_iter_cache_.assign(N, IterCache());
    }

    const bool ext_on = params_.extrinsic_est_en;

    const Eigen::Matrix3d R_LI = ext_on ? s.offset_R_L_I : Lidar_R_wrt_IMU.cast<double>();
    const Eigen::Vector3d T_LI = ext_on ? s.offset_T_L_I : Lidar_T_wrt_IMU.cast<double>();

    const double plane_thr = params_.plane_threshold;
    const double match_s = params_.match_sqaured;
    const double laser_cov = params_.laser_point_cov;

    const auto kf_rot = s.rotation;
    const auto kf_pos = s.position;
    const auto kf_vel = s.velocity;
    const Eigen::Vector3d w = s.omg;
    tbb::enumerable_thread_specific<Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>> local_solver(
        [] { return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(); }
    );
    tbb::enumerable_thread_specific<batch_measurement_result> local_results([] {
        return batch_measurement_result {};
    });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [&](const tbb::blocked_range<size_t>& r) {
        auto& solver = local_solver.local();
        auto& local_result = local_results.local();

        for (size_t i = r.begin(); i != r.end(); ++i) {
            const auto& p = current_batch.points[i];
            if (p.count < 1) {
                continue;
            }
            const double dt = p.timestamp - current_batch.timestamp;
            const Eigen::Vector3d pt_imu_d = R_LI * p.position.cast<double>() + T_LI;

            Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();

            R_delta = exp<double>(w * dt);

            // 先将每个点去畸变到当前批次时间，再搜索局部地图平面。
            const Eigen::Vector3d pt_odom_d = (kf_rot * R_delta) * pt_imu_d + kf_pos + kf_vel * dt;

            points_odom_frame[i] = pt_odom_d.cast<float>();
            const SmallOctVox::PositionIndex voxel_idx =
                ivox->get_position_index(points_odom_frame[i]);

            Eigen::Vector3d n;
            double d_plane = 0.0;
            auto& iter_cache = batch_iter_cache_[i];
            if (s.batch_iter > 0 && iter_cache.valid
                && (voxel_idx.array() == iter_cache.voxel_index.array()).all())
            {
                n = iter_cache.normal;
                d_plane = iter_cache.plane_d;
            } else {
                iter_cache.voxel_index = voxel_idx;
                iter_cache.valid = false;

                std::vector<Eigen::Vector3f> near;
                ivox->get_closest_point(points_odom_frame[i], near, NUM_MATCH_POINTS);
                if (near.size() < MIN_MATCH_POINTS) {
                    continue;
                }

                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                for (const auto& np: near) {
                    centroid += np.cast<double>();
                }
                centroid /= static_cast<double>(near.size());

                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& np: near) {
                    const Eigen::Vector3d d = np.cast<double>() - centroid;
                    cov.noalias() += d * d.transpose();
                }

                if (near.size() <= 1) {
                    continue;
                }
                cov /= static_cast<double>(near.size() - 1);

                // 邻域协方差矩阵的最小特征向量即拟合平面的法向量。
                solver.compute(cov);
                n = solver.eigenvectors().col(0);
                d_plane = -n.dot(centroid);

                if (s.batch_iter == 0) {
                    const double pt_norm = current_batch.points[i].position.norm();

                    // 首轮迭代先过滤弱匹配；后续迭代基于更精确的状态，
                    // 可以更直接地使用残差。
                    const double d_signed_first = n.dot(pt_odom_d) + d_plane;
                    if (pt_norm <= match_s * d_signed_first * d_signed_first) {
                        continue;
                    }

                    bool valid = true;
                    for (const auto& np: near) {
                        if (std::abs(n.dot(np.cast<double>()) + d_plane) > plane_thr) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) {
                        continue;
                    }
                }
                iter_cache.normal = n;
                iter_cache.plane_d = d_plane;
                iter_cache.valid = true;
            }

            const double d_signed = n.dot(pt_odom_d) + d_plane;

            const Eigen::Matrix<state::value_type, 3, 1> normal0 = n.cast<state::value_type>();
            Eigen::Matrix<state::value_type, 1, 12> H;

            if (ext_on) {
                // 开启外参估计时，在点到平面 Jacobian 的位姿项后追加
                // LiDAR-IMU 旋转和平移外参的导数。
                const Eigen::Matrix<state::value_type, 3, 1> C = s.rotation.transpose() * normal0;

                const Eigen::Matrix<state::value_type, 3, 1> A =
                    pt_imu_d.cast<state::value_type>().cross(C);

                const Eigen::Matrix<state::value_type, 3, 1> B =
                    point_lidar_frame.cast<state::value_type>().cross(
                        s.offset_R_L_I.transpose() * C
                    );

                H << normal0.transpose(), A.transpose(), B.transpose(), C.transpose();
            } else {
                const Eigen::Matrix<state::value_type, 3, 1> A =
                    pt_imu_d.cast<state::value_type>().cross(s.rotation.transpose() * normal0);

                H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            const state::value_type invR = static_cast<state::value_type>(1)
                / std::max(static_cast<state::value_type>(laser_cov),
                           static_cast<state::value_type>(1e-9));
            const state::value_type weight =
                invR * static_cast<state::value_type>(current_batch.points[i].count);
            local_result.HTRH.noalias() += H.transpose() * (H * weight);
            local_result.HTRz.noalias() += H.transpose() * (weight * -d_signed);
            ++local_result.effective_count;
        }
    });

    for (const auto& local_result: local_results) {
        result.HTRH.noalias() += local_result.HTRH;
        result.HTRz.noalias() += local_result.HTRz;
        result.effective_count += local_result.effective_count;
    }
}

void Estimator::h_point(const state& s, point_measurement_result& measurement_result) {
    measurement_result.valid = false;
    // 将当前 LiDAR 点转换到 IMU 再到里程计坐标系，用于在局部 iVox 地图中找邻域点。
    Eigen::Matrix<state::value_type, 3, 1> point_imu_frame;
    if (params_.extrinsic_est_en) {
        point_imu_frame =
            s.offset_R_L_I * point_lidar_frame.cast<state::value_type>() + s.offset_T_L_I;
    } else {
        point_imu_frame =
            Lidar_R_wrt_IMU * point_lidar_frame.cast<state::value_type>() + Lidar_T_wrt_IMU;
    }
    point_odom_frame = (s.rotation * point_imu_frame + s.position).cast<float>();
    ivox->get_closest_point(point_odom_frame, nearest_points, NUM_MATCH_POINTS);
    if (nearest_points.size() < MIN_MATCH_POINTS) {
        return;
    }
    // 对近邻点做 PCA 平面拟合，后续使用点到平面的距离作为滤波器量测。

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& p: nearest_points) {
        centroid.noalias() += p;
    }
    centroid /= static_cast<float>(nearest_points.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& p: nearest_points) {
        Eigen::Vector3f centered = p - centroid;
        covariance.noalias() += centered * centered.transpose();
    }
    covariance /= static_cast<float>(nearest_points.size() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    float d = -normal.dot(centroid);
    for (size_t j = 0; j < nearest_points.size(); j++) {
        float point_distanace = std::abs(normal.dot(nearest_points[j]) + d);
        if (point_distanace > params_.plane_threshold) {
            // 邻域点自身都不能很好落在同一平面时，说明该匹配不稳定，直接丢弃。
            return;
        }
    }
    float point_distanace = normal.dot(point_odom_frame) + d;
    if (point_lidar_frame.norm() <= params_.match_sqaured * point_distanace * point_distanace) {
        return;
    }
    // 计算点到平面残差及其对状态量的 Jacobian。
    measurement_result.laser_point_cov = static_cast<state::value_type>(params_.laser_point_cov);
    if (params_.extrinsic_est_en) {
        Eigen::Matrix<state::value_type, 3, 1> normal0 = normal.cast<state::value_type>();
        Eigen::Matrix<state::value_type, 3, 1> C = s.rotation.transpose() * normal0;
        Eigen::Matrix<state::value_type, 3, 1> A, B;
        A.noalias() = point_imu_frame.cross(C);
        B.noalias() =
            point_lidar_frame.cast<state::value_type>().cross(s.offset_R_L_I.transpose() * C);
        measurement_result.H << normal0.transpose(), A.transpose(), B.transpose(), C.transpose();
    } else {
        Eigen::Matrix<state::value_type, 3, 1> normal0 = normal.cast<state::value_type>();
        Eigen::Matrix<state::value_type, 3, 1> A;
        A.noalias() = point_imu_frame.cross(s.rotation.transpose() * normal0);
        measurement_result.H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }
    measurement_result.z = -point_distanace;
    measurement_result.valid = true;
}

void Estimator::h_imu(const state& s, imu_measurement_result& measurement_result) {
    std::memset(measurement_result.satu_check, false, 6);
    // IMU 量测模型约束角速度和线加速度，bias 作为状态量在滤波过程中估计。
    measurement_result.z.segment<3>(0) = angular_velocity - s.omg - s.bg;
    measurement_result.z.segment<3>(3) =
        linear_acceleration * imu_acceleration_scale - s.acceleration - s.ba;
    measurement_result.imu_meas_omg_cov = static_cast<state::value_type>(params_.imu_meas_omg_cov);
    measurement_result.imu_meas_acc_cov = static_cast<state::value_type>(params_.imu_meas_acc_cov);
    if (params_.check_satu) {
        for (int i = 0; i < 3; i++) {
            if (std::abs(angular_velocity(i)) >= params_.satu_gyro) {
                // 饱和轴的量测不可信，将残差置零，并通过 satu_check 告诉滤波器跳过该维。
                measurement_result.satu_check[i] = true;
                measurement_result.z(i) = 0.0;
            }
            if (std::abs(linear_acceleration(i)) >= params_.satu_acc) {
                // 加速度饱和同样不参与更新，避免剧烈碰撞或异常数据污染状态。
                measurement_result.satu_check[i + 3] = true;
                measurement_result.z(i + 3) = 0.0;
            }
        }
    }
}

} // namespace rose_nav::lm
