#include "lmpc.hpp"
#include "planner/common.hpp"
//clang-format off
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/src/Core/Matrix.h>
#include <memory>
#include <optional>
#include <rclcpp/logging.hpp>
//clang-format on
namespace rose_nav::planner {
struct LMPC::Impl {
    struct Params {
        double predict_dt = 0.1;
        int max_iter = 100;
        int predict_steps = 30;
        double max_speed = 2.0;
        double max_accel = 1.0;
        double delay_time = 0.0;
        double blind_radius;
        std::vector<double> Q = { 15.0, 15.0, 0.5, 0.5 };
        std::vector<double> Rd = { 1.0, 0.05 };
        std::vector<double> R = { 0.1, 0.1 };
        void load(const ParamsNode& config) {
            predict_dt = config.declare<double>("predict_dt");
            max_iter = config.declare<int>("max_iter");
            predict_steps = config.declare<int>("predict_steps");
            max_speed = config.declare<double>("max_speed");
            max_accel = config.declare<double>("max_accel");
            delay_time = config.declare<double>("delay_time");
            blind_radius = config.declare<double>("blind_radius");
            Q = config.declare<std::vector<double>>("Q");
            Rd = config.declare<std::vector<double>>("Rd");
            R = config.declare<std::vector<double>>("R");
        }
    } params_;
    Impl(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        rose_map_ = rose_map;
        params_.load(config);
        xref_ = Eigen::MatrixXd::Zero(4, params_.predict_steps + 1);
        dref_ = Eigen::MatrixXd::Zero(2, params_.predict_steps + 1);
        output_ = Eigen::MatrixXd::Zero(2, params_.predict_steps + 1);
        last_output_ = output_;
        xbar_.resize(params_.predict_steps + 1);
    }
    void set_traj(const Traj& traj) {
        if (traj.traj_pieces.size() > 1 && traj.get_traj_total_duration() > 0.01) {
            traj_ = traj;
            traj_duration_ = traj.get_traj_total_duration();
        }
    }
    void set_current(const RoboState& c) {
        now_state_w_ = c;
    }
    double get_esdf(const Eigen::Vector2d& pos) {
        return rose_map_->esdf()->get_esdf(pos.cast<float>());
    }
    bool get_re(const int T_in, double dt_in) {
    ws:
        P_.clear();
        P_.reserve(T_in);

        auto t_cur_and_dis = traj_.get_traj_time_by_pos(now_state_w_.pos);
        if (t_cur_and_dis.second > 0.5) {
            return false;
        }
        double t_cur = t_cur_and_dis.first;
        if (t_cur < 0.0)
            t_cur = 0.0;
        double t = t_cur + dt_in + params_.delay_time;

        Eigen::Vector2d pos_end = Eigen::Vector2d::Zero();
        Eigen::Vector2d vel_end = Eigen::Vector2d::Zero();
        Eigen::Vector2d acc_end = Eigen::Vector2d::Zero();
        Eigen::Vector2d now_pos = now_state_w_.pos;
        bool has_ref = false;
        double yaw_end = 0.0;
        double w_end = 0.0;

        for (int i = 0; i < T_in; ++i) {
            TrajPoint tp;

            if (t <= traj_duration_) {
                auto curr_pos = traj_.get_traj_pos_by_time(t);
                auto vec = curr_pos - now_pos;
                auto curr_vel = traj_.get_traj_vel_by_time(t);
                if (vec.norm() < params_.blind_radius) {
                    t += dt_in;
                    i--;
                    continue;
                }

                has_ref = true;
                tp.pos = curr_pos;
                tp.vel = curr_vel;
                tp.acc = traj_.get_traj_acc_by_time(t);
                double yaw = traj_.get_traj_yaw_by_time(t);
                yaw = yaw_end + angles::shortest_angular_distance(yaw_end, yaw);
                tp.yaw = yaw;
                tp.w = traj_.get_traj_yaw_dot_by_time(t);
                pos_end = tp.pos;
                vel_end = tp.vel;
                acc_end = tp.acc;
                yaw_end = tp.yaw;
                w_end = tp.w;

            } else {
                tp.pos = pos_end;
                tp.vel = Eigen::Vector2d::Zero();
                tp.acc = Eigen::Vector2d::Zero();
                tp.yaw = yaw_end;
                tp.w = w_end;
            }

            P_.push_back(tp);
            t += dt_in;
        }
        return has_ref;
    }
    void state_trans_omni(RoboState& s, double vx_cmd, double vy_cmd) {
        s.pos.x() += vx_cmd * params_.predict_dt;
        s.pos.y() += vy_cmd * params_.predict_dt;
        s.vel.x() = vx_cmd;
        s.vel.y() = vy_cmd;
    }
    void predict_motion() {
        xbar_[0] = now_state_w_;
        RoboState temp = now_state_w_;

        for (int i = 1; i <= params_.predict_steps; i++) {
            state_trans_omni(temp, output_(0, i - 1), output_(1, i - 1));
            xbar_[i] = temp;
        }
    }
    void solve_mpc() {
        const int steps = params_.predict_steps;
        const double dt = params_.predict_dt;

        if (steps <= 0) {
            std::cerr << "[MPC] steps invalid\n";
            return;
        }

        const int dimx = 2 * steps; // x y
        const int dimu = 2 * steps; // vx vy
        const int nx = dimx + dimu;

        qp_gradient_.setZero(nx);

        const auto Q = params_.Q;
        const auto R = params_.R;
        const auto Rd = params_.Rd;

        /* ---------------- cost gradient ---------------- */

        for (int i = 0; i < steps; ++i) {
            int xi = 2 * i;

            qp_gradient_[xi + 0] = -2.0 * Q[0] * xref_(0, i);
            qp_gradient_[xi + 1] = -2.0 * Q[1] * xref_(1, i);
        }

        for (int i = 0; i < steps; ++i) {
            int ui = dimx + 2 * i;

            qp_gradient_[ui + 0] = -2.0 * R[0] * dref_(0, i);
            qp_gradient_[ui + 1] = -2.0 * R[1] * dref_(1, i);
        }

        /* ---------------- Hessian ---------------- */

        std::vector<Eigen::Triplet<double>> H_trip;
        H_trip.reserve(nx * 4);

        /* state cost */

        for (int i = 0; i < steps; ++i) {
            int xi = 2 * i;

            if (i != 0) {
                H_trip.emplace_back(xi + 0, xi + 0, 2.0 * Q[0]);
                H_trip.emplace_back(xi + 1, xi + 1, 2.0 * Q[1]);
            }
        }

        /* control cost */

        for (int i = 0; i < steps; ++i) {
            int ui = dimx + 2 * i;

            double w0 = R[0];
            double w1 = R[1];

            if (i == 0 || i == steps - 1) {
                w0 += Rd[0];
                w1 += Rd[1];
            } else {
                w0 += 2.0 * Rd[0];
                w1 += 2.0 * Rd[1];
            }

            H_trip.emplace_back(ui, ui, 2.0 * w0);
            H_trip.emplace_back(ui + 1, ui + 1, 2.0 * w1);
        }

        /* velocity smooth */

        for (int i = 1; i < steps; ++i) {
            int ui = dimx + 2 * i;
            int up = dimx + 2 * (i - 1);

            H_trip.emplace_back(ui, up, -2.0 * Rd[0]);
            H_trip.emplace_back(up, ui, -2.0 * Rd[0]);

            H_trip.emplace_back(ui + 1, up + 1, -2.0 * Rd[1]);
            H_trip.emplace_back(up + 1, ui + 1, -2.0 * Rd[1]);
        }

        qp_hessian_.resize(nx, nx);
        qp_hessian_.setFromTriplets(H_trip.begin(), H_trip.end());
        qp_hessian_.makeCompressed();

        /* ---------------- constraints ---------------- */

        const int dyn_rows = 2 * steps;
        const int speed_rows = 2 * steps;
        const int smooth_rows = 2 * (steps - 1);

        const int nc = dyn_rows + speed_rows + smooth_rows;

        qp_lowerBound_.setConstant(nc, -1e10);
        qp_upperBound_.setConstant(nc, 1e10);

        /* initial state */

        Eigen::Vector2d x0;
        x0 << now_state_w_.pos.x(), now_state_w_.pos.y();

        qp_lowerBound_.segment(0, 2) = x0;
        qp_upperBound_.segment(0, 2) = x0;

        /* dynamics rows */

        for (int i = 1; i < steps; ++i) {
            int r = 2 * i;

            qp_lowerBound_[r + 0] = 0.0;
            qp_upperBound_[r + 0] = 0.0;

            qp_lowerBound_[r + 1] = 0.0;
            qp_upperBound_[r + 1] = 0.0;
        }

        /* velocity bounds */

        const double max_speed = params_.max_speed;

        int offset = dyn_rows;

        for (int i = 0; i < steps; ++i) {
            int r = offset + 2 * i;

            qp_lowerBound_[r + 0] = -max_speed;
            qp_upperBound_[r + 0] = max_speed;

            qp_lowerBound_[r + 1] = -max_speed;
            qp_upperBound_[r + 1] = max_speed;
        }

        /* velocity change */

        offset += speed_rows;

        const double max_dv = params_.max_accel * dt;

        for (int i = 1; i < steps; ++i) {
            int r = offset + 2 * (i - 1);

            qp_lowerBound_[r + 0] = -max_dv;
            qp_upperBound_[r + 0] = max_dv;

            qp_lowerBound_[r + 1] = -max_dv;
            qp_upperBound_[r + 1] = max_dv;
        }

        /* ---------------- A matrix ---------------- */

        if (!solver_initialized_) {
            Eigen::SparseMatrix<double> A(nc, nx);
            std::vector<Eigen::Triplet<double>> A_trip;

            /* initial state */

            A_trip.emplace_back(0, 0, 1.0);
            A_trip.emplace_back(1, 1, 1.0);

            /* dynamics */

            for (int i = 1; i < steps; ++i) {
                int r = 2 * i;

                int last = 2 * (i - 1);
                int ucol = dimx + 2 * (i - 1);

                A_trip.emplace_back(r, r, 1.0);
                A_trip.emplace_back(r, last, -1.0);
                A_trip.emplace_back(r, ucol, -dt);

                A_trip.emplace_back(r + 1, r + 1, 1.0);
                A_trip.emplace_back(r + 1, last + 1, -1.0);
                A_trip.emplace_back(r + 1, ucol + 1, -dt);
            }

            /* velocity constraint */

            int offsetA = dyn_rows;

            for (int i = 0; i < steps; ++i) {
                int r = offsetA + 2 * i;
                int c = dimx + 2 * i;

                A_trip.emplace_back(r, c, 1.0);
                A_trip.emplace_back(r + 1, c + 1, 1.0);
            }

            /* velocity smooth */

            offsetA += speed_rows;

            for (int i = 1; i < steps; ++i) {
                int r = offsetA + 2 * (i - 1);

                int u = dimx + 2 * i;
                int up = dimx + 2 * (i - 1);

                A_trip.emplace_back(r, u, 1.0);
                A_trip.emplace_back(r, up, -1.0);

                A_trip.emplace_back(r + 1, u + 1, 1.0);
                A_trip.emplace_back(r + 1, up + 1, -1.0);
            }

            A.setFromTriplets(A_trip.begin(), A_trip.end());
            A.makeCompressed();

            A_cache_ = A;

            solver_.settings()->setWarmStart(true);
            solver_.settings()->setAdaptiveRho(true);
            solver_.settings()->setMaxIteration(2000);
            solver_.settings()->setAbsoluteTolerance(1e-4);
            solver_.settings()->setRelativeTolerance(1e-4);
            solver_.settings()->setVerbosity(false);

            solver_.data()->setNumberOfVariables(nx);
            solver_.data()->setNumberOfConstraints(nc);

            solver_.data()->setLinearConstraintsMatrix(A_cache_);
            solver_.data()->setHessianMatrix(qp_hessian_);
            solver_.data()->setGradient(qp_gradient_);
            solver_.data()->setLowerBound(qp_lowerBound_);
            solver_.data()->setUpperBound(qp_upperBound_);

            if (!solver_.initSolver()) {
                std::cerr << "[OSQP] init failed\n";
                return;
            }

            solver_initialized_ = true;
        }

        solver_.updateHessianMatrix(qp_hessian_);
        solver_.updateGradient(qp_gradient_);
        solver_.updateBounds(qp_lowerBound_, qp_upperBound_);

        solver_.solveProblem();

        Eigen::VectorXd sol = solver_.getSolution();

        if (!sol.allFinite()) {
            std::cerr << "[OSQP] solution invalid\n";
            return;
        }

        /* output velocity */

        for (int i = 0; i < steps; ++i) {
            int ui = dimx + 2 * i;

            output_(0, i) = sol[ui + 0];
            output_(1, i) = sol[ui + 1];
        }
    }

    std::optional<ControlOutput> solve(std::chrono::duration<double> dt) {
        P_.clear();
        if (!get_re(params_.predict_steps, params_.predict_dt)) {
            RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "get_ref failed");
            return std::nullopt;
        }
        for (int i = 0; i < P_.size(); i++) {
            xref_(0, i) = P_[i].pos.x();
            xref_(1, i) = P_[i].pos.y();
            xref_(2, i) = P_[i].vel.x();
            xref_(3, i) = P_[i].vel.y();
            dref_(0, i) = P_[i].vel.x();
            dref_(1, i) = P_[i].vel.y();
        }
        const auto t_start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < params_.max_iter; ++iter) {
            const auto t_now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double>(t_now - t_start);

            if (elapsed >= dt) {
                break;
            }
            if (iter > 0) {
                double du = 0.0;
                for (int c = 0; c < output_.cols(); ++c) {
                    du += std::fabs(output_(0, c) - last_output_(0, c));
                    du += std::fabs(output_(1, c) - last_output_(1, c));
                }
                if (du < 1e-4) {
                    break;
                }
            }

            predict_motion();
            last_output_ = output_;
            solve_mpc();
        }
        ControlOutput o;
        o.vel = Eigen::Vector2d(output_(0, 0), output_(1, 0));
        o.pred_states = xbar_;
        return std::make_optional(o);
    }
    double traj_duration_;
    Traj traj_;
    RoboState now_state_w_;
    std::vector<RoboState> xbar_;
    Eigen::MatrixXd A_, B_;
    Eigen::VectorXd C_;
    Eigen::MatrixXd xref_;
    Eigen::MatrixXd dref_;
    Eigen::MatrixXd output_;
    Eigen::MatrixXd last_output_;
    Eigen::VectorXd last_solution_;
    bool has_last_solution_ = false;
    std::vector<TrajPoint> P_;
    OsqpEigen::Solver solver_;
    Eigen::VectorXd qp_gradient_;
    Eigen::VectorXd qp_lowerBound_;
    Eigen::VectorXd qp_upperBound_;
    Eigen::SparseMatrix<double> qp_hessian_;
    Eigen::SparseMatrix<double> A_cache_;
    int qp_nx_ = 0;
    int qp_nc_ = 0;
    bool solver_initialized_ = false;
    int A_rows_cached_ = 0;
    int A_cols_cached_ = 0;
    map::RoseMap::Ptr rose_map_;
};
LMPC::LMPC(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
    _impl = std::make_unique<Impl>(rose_map, config);
}
LMPC::~LMPC() {
    _impl.reset();
}
std::optional<ControlOutput> LMPC::solve(std::chrono::duration<double> dt) {
    return _impl->solve(dt);
}
void LMPC::set_traj(const Traj& traj) {
    _impl->set_traj(traj);
}
void LMPC::set_current(const RoboState& c) {
    _impl->set_current(c);
}
} // namespace rose_nav::planner