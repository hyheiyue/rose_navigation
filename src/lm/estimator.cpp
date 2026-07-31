/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "estimator.h"
#include "lm/eskf.h"
#include "lm/small_ivox.h"
#include "lm/small_oct_vox.hpp"
#include <cmath>
#include <tbb/tbb.h>
namespace rose_nav::lm {
constexpr int NUM_MATCH_POINTS = 5;
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
    // ESKF 本体不直接依赖地图或传感器格式，三个量测模型以回调注入，便于在逐点更新、
    // 批量更新和 IMU 约束之间复用同一套预测/更新框架。
    kf.max_iter = params_.max_iter;
    reset();
}

void Estimator::reset() {
    ivox = std::make_shared<SmallOctVox>(params_.map_resolution, 1000000);
    batch_plane_cache_.clear();
    point_plane_cache_.clear();
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

void Estimator::cache_plane(
    PlaneCache& cache,
    const SmallOctVox::PositionIndex& voxel_idx,
    const Eigen::Vector3d& normal,
    double plane_d,
    bool valid
) {
    PlaneCache::accessor plane;
    cache.insert(plane, voxel_idx);
    if (!valid && plane->second.valid) {
        return;
    }
    plane->second.normal = normal;
    plane->second.plane_d = plane_d;
    plane->second.valid = valid;
}

void Estimator::h_batch(const state& s, batch_measurement_result& result) noexcept {
    const size_t N = current_batch.points.size();

    result.reset();

    points_odom_frame.assign(N, Eigen::Vector3f::Zero());
    if (s.batch_iter == 0) {
        batch_plane_cache_.clear();
    }

    using Scalar = state::value_type;

    const bool ext_on = params_.extrinsic_est_en;

    const Eigen::Matrix<Scalar, 3, 3> R_LI = ext_on ? s.offset_R_L_I : Lidar_R_wrt_IMU;
    const Eigen::Matrix<Scalar, 3, 1> T_LI = ext_on ? s.offset_T_L_I : Lidar_T_wrt_IMU;

    const double plane_thr = params_.plane_threshold;
    const double match_s = params_.match_sqaured;
    const double laser_cov = params_.laser_point_cov;

    const Eigen::Matrix<Scalar, 3, 3> kf_rot = s.rotation;
    const Eigen::Matrix<Scalar, 3, 1> kf_pos = s.position;
    const Eigen::Matrix<Scalar, 3, 1> kf_vel = s.velocity;
    const Eigen::Matrix<Scalar, 3, 1> w = s.omg;

    auto process_point = [&](size_t i,
                             Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>& solver,
                             batch_measurement_result& measurement,
                             std::vector<Eigen::Vector3f>& near) {
        const auto& p = current_batch.points[i];
        if (p.count < 1) {
            return;
        }
        const double dt = p.timestamp - current_batch.timestamp;
        const Scalar dt_state = static_cast<Scalar>(dt);
        const Eigen::Matrix<Scalar, 3, 1> pt_imu = R_LI * p.position.cast<Scalar>() + T_LI;

        Eigen::Matrix<Scalar, 3, 3> R_delta = Eigen::Matrix<Scalar, 3, 3>::Identity();

        R_delta = exp<Scalar>(w * dt_state);

        // 先用匀速模型将每个点去畸变到当前批次时间，再搜索局部地图平面。
        const Eigen::Matrix<Scalar, 3, 1> pt_imu_deskew = R_delta * pt_imu;
        const Eigen::Matrix<Scalar, 3, 1> pt_odom =
            kf_rot * pt_imu_deskew + kf_pos + kf_vel * dt_state;
        const Eigen::Vector3d pt_odom_d = pt_odom.template cast<double>();

        points_odom_frame[i] = pt_odom_d.cast<float>();
        const SmallOctVox::PositionIndex voxel_idx = ivox->get_position_index(points_odom_frame[i]);

        Eigen::Vector3d n;
        double d_plane = 0.0;

        PlaneCache::const_accessor cached_plane;
        // 可以缓存的前提是在本次更新结束才会将更新后的点加入ivox，也就是更新过程中ivox完全不变
        if (batch_plane_cache_.find(cached_plane, voxel_idx)) {
            if (!cached_plane->second.valid) {
                return;
            }
            n = cached_plane->second.normal;
            d_plane = cached_plane->second.plane_d;
        } else {
            cached_plane.release();

            near.clear();
            ivox->get_closest_point(points_odom_frame[i], near, NUM_MATCH_POINTS);
            if (near.size() < MIN_MATCH_POINTS) {
                cache_plane(batch_plane_cache_, voxel_idx, Eigen::Vector3d::Zero(), 0.0, false);
                return;
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
                return;
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
                    return;
                }

                bool valid = true;
                for (const auto& np: near) {
                    if (std::abs(n.dot(np.cast<double>()) + d_plane) > plane_thr) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) {
                    cache_plane(batch_plane_cache_, voxel_idx, Eigen::Vector3d::Zero(), 0.0, false);
                    return;
                }
            }

            // 同一批次内落在同一体素的点复用平面模型，减少近邻搜索和 PCA 次数。
            cache_plane(batch_plane_cache_, voxel_idx, n, d_plane, true);
        }

        const double d_signed = n.dot(pt_odom_d) + d_plane;

        const Eigen::Matrix<Scalar, 3, 1> normal0 = n.cast<Scalar>();
        Eigen::Matrix<Scalar, 1, batch_measurement_result::DIM> H;
        H.setZero();
        const Eigen::Matrix<Scalar, 3, 1> velocity_jac = normal0 * dt_state;

        if (ext_on) {
            // 开启外参估计时，在点到平面 Jacobian 的位姿项后追加
            // LiDAR-IMU 旋转和平移外参的导数。
            const Eigen::Matrix<Scalar, 3, 1> C = s.rotation.transpose() * normal0;
            const Eigen::Matrix<Scalar, 3, 1> C_deskew = R_delta.transpose() * C;

            const Eigen::Matrix<Scalar, 3, 1> A = pt_imu_deskew.cross(C);

            const Eigen::Matrix<Scalar, 3, 1> B =
                p.position.cast<Scalar>().cross(s.offset_R_L_I.transpose() * C_deskew);

            H.template segment<3>(state::position_index) = normal0.transpose();
            H.template segment<3>(state::rotation_index) = A.transpose();
            H.template segment<3>(state::offset_R_L_I_index) = B.transpose();
            H.template segment<3>(state::offset_T_L_I_index) = C_deskew.transpose();
        } else {
            const Eigen::Matrix<Scalar, 3, 1> A =
                pt_imu_deskew.cross(s.rotation.transpose() * normal0);

            H.template segment<3>(state::position_index) = normal0.transpose();
            H.template segment<3>(state::rotation_index) = A.transpose();
        }
        H.template segment<3>(state::velocity_index) = velocity_jac.transpose();

        const state::value_type invR = static_cast<state::value_type>(1)
            / std::max(static_cast<state::value_type>(laser_cov),
                       static_cast<state::value_type>(1e-9));
        const state::value_type weight =
            invR * static_cast<state::value_type>(current_batch.points[i].count);
        // const state::value_type weight = invR;
        // 直接累加 H^T R^-1 H 和 H^T R^-1 z，避免保存每个点的 Jacobian。
        // 这相当于把大规模点云量测压缩成固定维度正规方程，是批量更新的核心优化。
        measurement.HTRH.noalias() += H.transpose() * (H * weight);
        measurement.HTRz.noalias() += H.transpose() * (weight * -d_signed);
        ++measurement.effective_count;
    };

    if (params_.h_batch_parallel) {
        // 每个线程独立持有特征分解器和正规方程缓存，避免并行 PCA 时反复分配对象或抢锁。
        tbb::enumerable_thread_specific<Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>>
            local_solver([] { return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(); });
        tbb::enumerable_thread_specific<batch_measurement_result> local_results([] {
            return batch_measurement_result {};
        });
        tbb::enumerable_thread_specific<std::vector<Eigen::Vector3f>> local_near_points([] {
            std::vector<Eigen::Vector3f> near;
            near.reserve(NUM_MATCH_POINTS);
            return near;
        });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, N),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& solver = local_solver.local();
                auto& local_result = local_results.local();
                auto& near = local_near_points.local();

                for (size_t i = r.begin(); i != r.end(); ++i) {
                    process_point(i, solver, local_result, near);
                }
            }
        );

        for (const auto& local_result: local_results) {
            result.HTRH.noalias() += local_result.HTRH;
            result.HTRz.noalias() += local_result.HTRz;
            result.effective_count += local_result.effective_count;
        }
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        std::vector<Eigen::Vector3f> near;
        near.reserve(NUM_MATCH_POINTS);

        for (size_t i = 0; i < N; ++i) {
            process_point(i, solver, result, near);
        }
    }
}

void Estimator::h_point(const state& s, point_measurement_result& measurement_result) {
    using Scalar = state::value_type;

    measurement_result.valid = false;
    // 将当前 LiDAR 点转换到 IMU 再到里程计坐标系，用于在局部 iVox 地图中找邻域点。
    Eigen::Matrix<Scalar, 3, 1> point_imu_frame;
    if (params_.extrinsic_est_en) {
        point_imu_frame = s.offset_R_L_I * point_lidar_frame.cast<Scalar>() + s.offset_T_L_I;
    } else {
        point_imu_frame = Lidar_R_wrt_IMU * point_lidar_frame.cast<Scalar>() + Lidar_T_wrt_IMU;
    }
    const Eigen::Matrix<Scalar, 3, 1> point_odom = s.rotation * point_imu_frame + s.position;
    point_odom_frame = point_odom.template cast<float>();
    // 单点模式以当前状态预测后的 odom 体素为缓存键；如果该体素已经拟合过稳定平面，
    // 后续点或迭代重线性化可直接复用，避免重复近邻搜索和 PCA。
    // 可以缓存的前提是在本次更新结束才会将更新后的点加入ivox，也就是更新过程中ivox完全不变
    const SmallOctVox::PositionIndex voxel_idx = ivox->get_position_index(point_odom_frame);
    Eigen::Vector3d normal_d;
    double plane_d = 0.0;
    bool can_reuse_plane = false;

    PlaneCache::const_accessor cached_plane;
    if (point_plane_cache_.find(cached_plane, voxel_idx)) {
        if (!cached_plane->second.valid) {
            return;
        }
        normal_d = cached_plane->second.normal;
        plane_d = cached_plane->second.plane_d;
        const double cached_dist = normal_d.dot(point_odom_frame.cast<double>()) + plane_d;
        // 缓存平面必须仍能解释当前点；残差过大说明状态变化或地图更新后平面已不适用。
        can_reuse_plane =
            std::isfinite(cached_dist) && std::abs(cached_dist) <= params_.plane_threshold;
        cached_plane.release();
    }

    if (!can_reuse_plane) {
        ivox->get_closest_point(point_odom_frame, nearest_points, NUM_MATCH_POINTS);
        if (nearest_points.size() < MIN_MATCH_POINTS) {
            cache_plane(point_plane_cache_, voxel_idx, Eigen::Vector3d::Zero(), 0.0, false);
            return;
        }
        // 对近邻点做 PCA 平面拟合，后续使用点到平面的距离作为滤波器量测。
        // 相比点到点残差，点到平面残差对低线数/非重复扫描 LiDAR 更稳定，收敛也更快。

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& p: nearest_points) {
            centroid.noalias() += p.cast<double>();
        }
        centroid /= static_cast<double>(nearest_points.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto& p: nearest_points) {
            Eigen::Vector3d centered = p.cast<double>() - centroid;
            covariance.noalias() += centered * centered.transpose();
        }
        covariance /= static_cast<double>(nearest_points.size() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        normal_d = solver.eigenvectors().col(0);
        plane_d = -normal_d.dot(centroid);
        for (const auto& nearest_point: nearest_points) {
            const double nearest_dist =
                std::abs(normal_d.dot(nearest_point.cast<double>()) + plane_d);
            if (nearest_dist > params_.plane_threshold) {
                // 邻域点自身都不能很好落在同一平面时，说明该匹配不稳定，直接丢弃。
                cache_plane(point_plane_cache_, voxel_idx, Eigen::Vector3d::Zero(), 0.0, false);
                return;
            }
        }

        cache_plane(point_plane_cache_, voxel_idx, normal_d, plane_d, true);
    }

    const double point_distanace = normal_d.dot(point_odom_frame.cast<double>()) + plane_d;
    if (point_lidar_frame.norm() <= params_.match_sqaured * point_distanace * point_distanace) {
        return;
    }
    // 计算点到平面残差及其对状态量的 Jacobian。
    measurement_result.laser_point_cov = static_cast<Scalar>(params_.laser_point_cov);
    if (params_.extrinsic_est_en) {
        Eigen::Matrix<Scalar, 3, 1> normal0 = normal_d.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> C = s.rotation.transpose() * normal0;
        Eigen::Matrix<Scalar, 3, 1> A, B;
        A.noalias() = point_imu_frame.cross(C);
        B.noalias() = point_lidar_frame.cast<Scalar>().cross(s.offset_R_L_I.transpose() * C);
        // 外参在线估计时，H 同时包含车体位姿和 LiDAR-IMU 外参的扰动项。
        measurement_result.H << normal0.transpose(), A.transpose(), B.transpose(), C.transpose();
    } else {
        Eigen::Matrix<Scalar, 3, 1> normal0 = normal_d.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> A;
        A.noalias() = point_imu_frame.cross(s.rotation.transpose() * normal0);
        measurement_result.H << normal0.transpose(), A.transpose(), static_cast<Scalar>(0.0),
            static_cast<Scalar>(0.0), static_cast<Scalar>(0.0), static_cast<Scalar>(0.0),
            static_cast<Scalar>(0.0), static_cast<Scalar>(0.0);
    }
    measurement_result.z = -static_cast<Scalar>(point_distanace);
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
