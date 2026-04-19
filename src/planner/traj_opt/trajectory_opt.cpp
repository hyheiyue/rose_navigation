#include "trajectory_opt.hpp"
#include "lbfgs.hpp"
#include "planner/common.hpp"
#include "planner/traj.hpp"
#include "planner/traj_opt/minco.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <utility>
#include <vector>
namespace rose_nav::planner {
struct TrajOpt::Impl {
    struct Params {
        double smooth_weight;
        double obstacle_weight;
        bool enable;
        void load(const ParamsNode& config) {
            smooth_weight = config.declare<double>("smooth_weight");
            obstacle_weight = config.declare<double>("obstacle_weight");
            enable = config.declare<bool>("enable");
        }
    } params_;
    Impl(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        rose_map_ = rose_map;
        params_.load(config);
        lbfgs_params_.mem_size = 256;
        lbfgs_params_.past = 20;
        lbfgs_params_.min_step = 1e-32;
        lbfgs_params_.g_epsilon = 2.0e-7;
        lbfgs_params_.delta = 2e-7;
        lbfgs_params_.max_iterations = 4000;
        lbfgs_params_.max_linesearch = 32;
        lbfgs_params_.f_dec_coeff = 1e-4;
        lbfgs_params_.s_curv_coeff = 0.9;
    }
    static inline bool
    smoothed_l1(const double& x, const double& mu, double& f, double& df) noexcept {
        if (x < 0.0) {
            return false;
        } else if (x > mu) {
            f = x - 0.5 * mu;
            df = 1.0;
            return true;
        } else {
            const double xdmu = x / mu;
            const double sqrxdmu = xdmu * xdmu;
            const double mumxd2 = mu - 0.5 * x;
            f = mumxd2 * sqrxdmu * xdmu;
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return true;
        }
    }
    static inline double kahan_sum(double& sum, double& c, const double& val) noexcept {
        double y = val - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
        return sum;
    }
    double attach_penalty_functional(const Eigen::Matrix2Xd& in_ps, Eigen::Matrix2Xd& gradp)
        const noexcept {
        const int N = in_ps.cols();
        if (N < 2)
            return 0.0;

        double cost_val = 0.0;
        double c_cost = 0.0;

        for (int i = 0; i < N - 1; i++) {
            double nearest_cost = 0.0;
            const Eigen::Vector2d& p0 = in_ps.col(i);
            const Eigen::Vector2d& p1 = in_ps.col(i + 1);

            // 1. Obstacle cost
            Eigen::Vector2d obs_grad = obstacle_term(p0, nearest_cost);
            kahan_sum(cost_val, c_cost, nearest_cost);
            gradp.col(i).noalias() += obs_grad;
        }

        return cost_val;
    }
    Eigen::Vector2d
    obstacle_term(const Eigen::Vector2d& xcur, double& nearest_cost) const noexcept {
        nearest_cost = 0.0;
        Eigen::Vector2d grad = Eigen::Vector2d::Zero();

        const double R = 1.0;
        const double mu = 0.4;

        double d;
        Eigen::Vector2d g;
        if (!sample_esdf_and_grad(xcur, d, g) || !std::isfinite(d)) {
            return grad;
        }

        if (d > R) {
            return grad;
        }

        double penetration = R - d;
        double cost_s1 = 0.0, dcost_s1 = 0.0;
        if (!smoothed_l1(penetration, mu, cost_s1, dcost_s1)) {
            return grad;
        }

        const auto w_obs = params_.obstacle_weight;
        nearest_cost = w_obs * cost_s1;
        Eigen::Vector2d dir = (g.norm() > 1e-6) ? g.normalized() : Eigen::Vector2d::Zero();
        grad = w_obs * dcost_s1 * (-dir);

        if (!grad.allFinite()) {
            grad.setZero();
        }

        return grad;
    }

    bool
    sample_esdf_and_grad(const Eigen::Vector2d& pos, double& out_dist, Eigen::Vector2d& out_grad)
        const noexcept {
        if (!rose_map_)
            return false;
        const auto& esdf = rose_map_->esdf();
        const double vs = esdf->esdf_->voxel_size;
        if (vs <= 0.0)
            return false;

        const double R = 1.0;

        Eigen::Vector2f pos2f(pos.x(), pos.y());
        auto key = esdf->world_to_key(pos2f);
        if (esdf->key_to_index(key) < 0)
            return false;
        auto read = [&](int dx, int dy) {
            map::VoxelKey<2> k { key.x() + dx, key.y() + dy };
            int idx = esdf->key_to_index(k);
            if (idx < 0)
                return R + 1.0;

            double d = esdf->get_esdf(idx);
            if (!std::isfinite(d))
                return R + 1.0;

            return std::min(d, R + 1.0);
        };

        double d[4][4];
        for (int iy = 0; iy < 4; ++iy) {
            for (int ix = 0; ix < 4; ++ix) {
                d[iy][ix] = read(ix - 1, iy - 1);
            }
        }

        double fx = (pos2f.x() - (key.x() + 0.5) * vs) / vs + 0.5;
        double fy = (pos2f.y() - (key.y() + 0.5) * vs) / vs + 0.5;

        fx = std::clamp(fx, 0.0, 1.0);
        fy = std::clamp(fy, 0.0, 1.0);

        auto w = [](double t, int i) {
            // i = 0,1,2,3  →  p_{-1}, p_0, p_1, p_2
            double x = std::abs(t - (i - 1));
            if (x < 1.0)
                return 1.5 * x * x * x - 2.5 * x * x + 1.0;
            else if (x < 2.0)
                return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
            else
                return 0.0;
        };

        auto dw = [](double t, int i) {
            double s = t - (i - 1);
            double x = std::abs(s);
            double sign = (s >= 0.0) ? 1.0 : -1.0;

            if (x < 1.0)
                return sign * (4.5 * x * x - 5.0 * x);
            else if (x < 2.0)
                return sign * (-1.5 * x * x + 5.0 * x - 4.0);
            else
                return 0.0;
        };

        double val = 0.0;
        double gx = 0.0;
        double gy = 0.0;

        for (int iy = 0; iy < 4; ++iy) {
            double wy = w(fy, iy);
            double dwy = dw(fy, iy);

            for (int ix = 0; ix < 4; ++ix) {
                double wx = w(fx, ix);
                double dwx = dw(fx, ix);

                double v = d[iy][ix];
                val += v * wx * wy;
                gx += v * dwx * wy;
                gy += v * wx * dwy;
            }
        }

        out_dist = val;
        out_grad = Eigen::Vector2d(gx / vs, gy / vs);

        double gnorm = out_grad.norm();
        if (gnorm > 1e-6) {
            out_grad *= std::min(1.0, 1.0 / gnorm);
        }
        if (out_dist > R) {
            out_grad.setZero();
        }

        return out_dist < R;
    }

    std::vector<Piece<5, 2>> optimize(
        const std::vector<Traj::SampledPoint>& sampled,
        double dt,
        std::optional<std::pair<int, int>> some_no_opt = std::nullopt
    ) {
        if (sampled.size() < 5) {
            RCLCPP_ERROR_STREAM(
                rclcpp::get_logger("rose_nav:planner"),
                "sampled too small size: " << sampled.size()
            );
            return {};
        }
        int no_opt_start = 0;
        int no_opt_end = 0;

        if (some_no_opt) {
            no_opt_start = some_no_opt->first;
            no_opt_end = some_no_opt->second;
            no_opt_start = std::max(0, no_opt_start);
            no_opt_end = std::min(static_cast<int>(sampled.size()) - 2, no_opt_end);
        }
        sampled_path_ = sampled;

        Eigen::Matrix<double, 2, 3> head_state;
        head_state << sampled_path_.front().p, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero();

        Eigen::Matrix<double, 2, 3> tail_state;
        tail_state << sampled_path_.back().p, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero();

        const int piece_num = sampled_path_.size() - 1;
        piece_num_ = piece_num;

        const auto w_smooth = params_.smooth_weight;

        minco_
            .setConditions(head_state, tail_state, piece_num, Eigen::Vector2d(w_smooth, w_smooth));

        const int ctrl_num = piece_num - 1;
        std::vector<char> is_no_opt(ctrl_num, false);

        if (some_no_opt) {
            no_opt_end = std::clamp(no_opt_end, 0, ctrl_num);
            no_opt_start = std::clamp(no_opt_start, 0, ctrl_num);
            if (no_opt_start != no_opt_end) {
                for (int i = no_opt_start; i < no_opt_end; ++i) {
                    is_no_opt[i] = true;
                }
            }
        }

        opt_indices_.clear();
        for (int i = 0; i < ctrl_num; ++i) {
            if (!is_no_opt[i]) {
                opt_indices_.push_back(i);
            }
        }
        const int opt_num = opt_indices_.size();

        Eigen::VectorXd x_opt(2 * opt_num);

        for (int k = 0; k < opt_num; ++k) {
            int i = opt_indices_[k];
            x_opt(k) = sampled_path_[i + 1].p.x();
            x_opt(k + opt_num) = sampled_path_[i + 1].p.y();
        }

        in_times_.resize(piece_num);
        for (int i = 0; i < piece_num; ++i) {
            in_times_(i) = dt;
        }

        double min_cost = 0.0;
        int ret = 0;

        if (opt_num > 0 && params_.enable) {
            ret = lbfgs::lbfgs_optimize(
                x_opt,
                min_cost,
                &TrajOpt::Impl::cost,
                nullptr,
                nullptr,
                this,
                lbfgs_params_
            );
        }

        Eigen::Matrix2Xd in_ps(2, ctrl_num);
        Eigen::VectorXd full_x(2 * ctrl_num);

        for (int i = 0; i < ctrl_num; ++i) {
            full_x(i) = sampled_path_[i + 1].p.x();
            full_x(i + ctrl_num) = sampled_path_[i + 1].p.y();
        }

        for (int k = 0; k < opt_num; ++k) {
            int i = opt_indices_[k];
            full_x(i) = x_opt(k);
            full_x(i + ctrl_num) = x_opt(k + opt_num);
        }

        in_ps.row(0) = full_x.head(ctrl_num).transpose();
        in_ps.row(1) = full_x.segment(ctrl_num, ctrl_num).transpose();

        minco_.setParameters(in_ps, in_times_);

        std::vector<Piece<5, 2>> final_traj;
        minco_.getPieces(final_traj);

        if (!(ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMLINESEARCH)) {
            RCLCPP_ERROR_STREAM(
                rclcpp::get_logger("rose_nav:planner"),
                "TrajOpt FAIL: " << lbfgs::lbfgs_strerror(ret)
            );
        }

        return final_traj;
    }

    static double cost(void* ptr, const Eigen::VectorXd& x, Eigen::VectorXd& g) noexcept {
        auto* instance = static_cast<TrajOpt::Impl*>(ptr);

        const int ctrl_num = instance->piece_num_ - 1;
        const int opt_num = instance->opt_indices_.size();

        double cost_val = 0.0;

        Eigen::VectorXd full_x(2 * ctrl_num);

        for (int i = 0; i < ctrl_num; ++i) {
            full_x(i) = instance->sampled_path_[i + 1].p.x();
            full_x(i + ctrl_num) = instance->sampled_path_[i + 1].p.y();
        }

        for (int k = 0; k < opt_num; ++k) {
            int i = instance->opt_indices_[k];
            full_x(i) = x(k);
            full_x(i + ctrl_num) = x(k + opt_num);
        }

        Eigen::Matrix2Xd in_ps(2, ctrl_num);
        in_ps.row(0) = full_x.head(ctrl_num).transpose();
        in_ps.row(1) = full_x.segment(ctrl_num, ctrl_num).transpose();

        instance->minco_.setParameters(in_ps, instance->in_times_);

        double energy = 0.0;

        Eigen::Matrix2Xd energy_grad = Eigen::Matrix2Xd::Zero(2, ctrl_num);
        Eigen::VectorXd energyT_grad = Eigen::VectorXd::Zero(ctrl_num + 1);

        Eigen::Matrix2Xd grad_by_points;
        Eigen::VectorXd grad_by_times;
        Eigen::MatrixX2d partial_grad_by_coeffs;
        Eigen::VectorXd partial_grad_by_times;

        instance->minco_.getEnergyPartialGradByCoeffs(partial_grad_by_coeffs);
        instance->minco_.getEnergyPartialGradByTimes(partial_grad_by_times);
        instance->minco_.getEnergy(energy);

        instance->minco_.propogateGrad(
            partial_grad_by_coeffs,
            partial_grad_by_times,
            energy_grad,
            energyT_grad
        );

        Eigen::Matrix2Xd gradp = energy_grad;

        cost_val += energy;
        cost_val += instance->attach_penalty_functional(in_ps, gradp);

        Eigen::VectorXd g_full(2 * ctrl_num);
        g_full.setZero();

        g_full.segment(0, ctrl_num) = gradp.row(0).transpose();
        g_full.segment(ctrl_num, ctrl_num) = gradp.row(1).transpose();

        g.resize(2 * opt_num);
        g.setZero();

        for (int k = 0; k < opt_num; ++k) {
            int i = instance->opt_indices_[k];
            g(k) = g_full(i);
            g(k + opt_num) = g_full(i + ctrl_num);
        }

        return cost_val;
    }

    int piece_num_;
    map::RoseMap::Ptr rose_map_;
    minco::MINCO_S3NU minco_;
    lbfgs::lbfgs_parameter_t lbfgs_params_;
    Eigen::VectorXd in_times_;
    std::vector<Traj::SampledPoint> sampled_path_;
    std::vector<int> opt_indices_;
};
TrajOpt::TrajOpt(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
    _impl = std::make_unique<Impl>(rose_map, config);
}
TrajOpt::~TrajOpt() {
    _impl.reset();
}
std::vector<Piece<5, 2>> TrajOpt::optimize(
    const std::vector<Traj::SampledPoint>& sampled,
    double dt,
    std::optional<std::pair<int, int>> some_no_opt
) {
    return _impl->optimize(sampled, dt,  some_no_opt);
}
} // namespace rose_nav::planner