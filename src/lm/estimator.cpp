/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "estimator.h"
#include <tbb/tbb.h>
namespace rose_nav::lm {
constexpr int NUM_MATCH_POINTS = 100;
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
    reset();
}

void Estimator::reset() {
    ivox = std::make_shared<SmallIVox>(params_.map_resolution, 1000000);
    kf.P = Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Identity() * 0.01;
    kf.P.block<3, 3>(state::gravity_index, state::gravity_index).diagonal().fill(0.0001);
    kf.P.block<3, 3>(state::bg_index, state::bg_index).diagonal().fill(0.001);
    kf.P.block<3, 3>(state::ba_index, state::ba_index).diagonal().fill(0.001);
}

[[nodiscard]] Eigen::Matrix<state::value_type, state::DIM, state::DIM>
Estimator::process_noise_cov() const {
    Eigen::Matrix<state::value_type, state::DIM, state::DIM> cov =
        Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Zero();
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

void Estimator::h_batch(const state& s, std::vector<point_measurement_result>& results) noexcept {
    const size_t N = current_batch.points.size();

    results.clear();
    results.resize(N);

    points_odom_frame.assign(N, Eigen::Vector3f::Zero());

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

    tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [&](const tbb::blocked_range<size_t>& r) {
        auto& solver = local_solver.local();

        for (size_t i = r.begin(); i != r.end(); ++i) {
            const auto& p = current_batch.points[i];
            const double dt = p.timestamp - current_batch.timestamp;
            const Eigen::Vector3d pt_imu_d = R_LI * p.position.cast<double>() + T_LI;

            Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
            if (w.norm() > 1e-8) {
                R_delta = exp<double>(w * dt);
            }

            const Eigen::Vector3d pt_odom_d = (kf_rot * R_delta) * pt_imu_d + kf_pos + kf_vel * dt;

            points_odom_frame[i] = pt_odom_d.cast<float>();

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

            solver.compute(cov);
            const Eigen::Vector3d n = solver.eigenvectors().col(0);
            const double d_plane = -n.dot(centroid);

            const double d_signed = n.dot(pt_odom_d) + d_plane;

            if (s.batch_iter == 0) {
                const double pt_norm = current_batch.points[i].position.norm();

                if (pt_norm <= match_s * d_signed * d_signed) {
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

            point_measurement_result mr {};
            mr.valid = true;
            mr.z = -d_signed;
            mr.laser_point_cov = laser_cov;
            mr.count = current_batch.points[i].count;

            const Eigen::Matrix<state::value_type, 3, 1> normal0 = n.cast<state::value_type>();

            if (ext_on) {
                const Eigen::Matrix<state::value_type, 3, 1> C = s.rotation.transpose() * normal0;

                const Eigen::Matrix<state::value_type, 3, 1> A =
                    pt_imu_d.cast<state::value_type>().cross(C);

                const Eigen::Matrix<state::value_type, 3, 1> B =
                    point_lidar_frame.cast<state::value_type>().cross(
                        s.offset_R_L_I.transpose() * C
                    );

                mr.H << normal0.transpose(), A.transpose(), B.transpose(), C.transpose();
            } else {
                const Eigen::Matrix<state::value_type, 3, 1> A =
                    pt_imu_d.cast<state::value_type>().cross(s.rotation.transpose() * normal0);

                mr.H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            results[i] = mr;
        }
    });
}

void Estimator::h_point(const state& s, point_measurement_result& measurement_result) {
    measurement_result.valid = false;
    // get closest point
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
    // estimate plane

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
            return;
        }
    }
    float point_distanace = normal.dot(point_odom_frame) + d;
    if (point_lidar_frame.norm() <= params_.match_sqaured * point_distanace * point_distanace) {
        return;
    }
    // calculate residual and jacobian matrix
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
    measurement_result.z.segment<3>(0) = angular_velocity - s.omg - s.bg;
    measurement_result.z.segment<3>(3) =
        linear_acceleration * imu_acceleration_scale - s.acceleration - s.ba;
    measurement_result.imu_meas_omg_cov = static_cast<state::value_type>(params_.imu_meas_omg_cov);
    measurement_result.imu_meas_acc_cov = static_cast<state::value_type>(params_.imu_meas_acc_cov);
    if (params_.check_satu) {
        for (int i = 0; i < 3; i++) {
            if (std::abs(angular_velocity(i)) >= params_.satu_gyro) {
                measurement_result.satu_check[i] = true;
                measurement_result.z(i) = 0.0;
            }
            if (std::abs(linear_acceleration(i)) >= params_.satu_acc) {
                measurement_result.satu_check[i + 3] = true;
                measurement_result.z(i + 3) = 0.0;
            }
        }
    }
}

} // namespace rose_nav::lm
