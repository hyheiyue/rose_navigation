#pragma once
#include "traj_opt/trajectory.hpp"
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace rose_nav {

class Traj {
public:
    struct SampledPoint {
        Eigen::Vector2d p; // 采样点
        int raw_idx0; // 插值左端
        int raw_idx1; // 插值右端
        double t; // 插值比例
        double s; // 弧长位置
    };

public:
    inline void set_raw_path(const std::vector<Eigen::Vector2d>& path) {
        raw_path_ = path;
        raw_s_ = compute_arc_lengths(raw_path_);
    }

    inline void resample(double sample_ds) {
        if (raw_path_.size() < 2)
            return;

        const double s_total = raw_s_.back();

        sampled_.clear();
        raw_to_sampled_.clear();
        raw_to_sampled_.resize(raw_path_.size());

        for (double s = 0.0; s <= s_total; s += sample_ds) {
            auto sp = interpolate_with_map(raw_path_, raw_s_, s);
            int idx = sampled_.size();
            sampled_.push_back(sp);
            raw_to_sampled_[sp.raw_idx0].push_back(idx);
            if (sp.raw_idx1 != sp.raw_idx0)
                raw_to_sampled_[sp.raw_idx1].push_back(idx);
        }

        if (sampled_.empty() || sampled_.back().s < s_total) {
            int n = static_cast<int>(raw_path_.size()) - 1;
            SampledPoint sp { raw_path_.back(), n, n, 0.0, s_total };
            int idx = sampled_.size();
            sampled_.push_back(sp);
            raw_to_sampled_[n].push_back(idx);
        }
    }
    inline double get_traj_total_duration() const {
        int N = traj_pieces.size();
        double total_duration = 0.0;
        for (int i = 0; i < N; i++) {
            total_duration += traj_pieces[i].getDuration();
        }
        return total_duration;
    }
    inline std::pair<int, double> locate_traj_piece_idx_by_time(double t) const {
        int N = traj_pieces.size();
        int idx;
        double dur;
        for (idx = 0; idx < N && t > (dur = traj_pieces[idx].getDuration()); idx++) {
            t -= dur;
        }
        if (idx == N) {
            idx--;
            t += traj_pieces[idx].getDuration();
        }
        return std::make_pair(idx, t);
    }
    inline Eigen::VectorXd get_traj_pos_by_time(double t) const {
        auto [idx, t_in_piece] = locate_traj_piece_idx_by_time(t);
        return traj_pieces[idx].getPos(t_in_piece);
    }
    inline double get_traj_yaw_by_time(double t) const {
        auto [idx, t_in_piece] = locate_traj_piece_idx_by_time(t);
        return traj_pieces[idx].getYaw(t_in_piece);
    }
    inline double get_traj_yaw_dot_by_time(double t) const {
        auto [idx, t_in_piece] = locate_traj_piece_idx_by_time(t);
        return traj_pieces[idx].getYawDot(t_in_piece);
    }
    inline Eigen::VectorXd get_traj_vel_by_time(double t) const {
        auto [idx, t_in_piece] = locate_traj_piece_idx_by_time(t);
        return traj_pieces[idx].getVel(t_in_piece);
    }
    inline Eigen::VectorXd get_traj_acc_by_time(double t) const {
        auto [idx, t_in_piece] = locate_traj_piece_idx_by_time(t);
        return traj_pieces[idx].getAcc(t_in_piece);
    }
    inline std::pair<double, double>
    get_traj_time_by_pos(const Eigen::Vector2d& pos, double smaple_dt = 0.05) const {
        double best_t = 0.0;
        double t = 0.0;
        double total_duration = get_traj_total_duration();
        double min_dis = std::numeric_limits<double>::infinity();
        while (t <= total_duration) {
            auto p = get_traj_pos_by_time(t);
            auto dis = (p - pos).norm();
            if (dis < min_dis) {
                min_dis = dis;
                best_t = t;
            }
            t += smaple_dt;
        }
        return std::make_pair(best_t, min_dis);
    }

    // raw -> sampled（O(1)）
    inline const std::vector<int>& get_sampled_from_raw(int raw_idx) const noexcept {
        return raw_to_sampled_[raw_idx];
    }

    // 找某个 raw 点对应最近的 sampled 点（基于弧长）
    inline int find_nearest_sampled_of_raw(int raw_idx) const noexcept {
        double s_q = raw_s_[raw_idx];

        auto it = std::lower_bound(
            sampled_.begin(),
            sampled_.end(),
            s_q,
            [](const SampledPoint& sp, double val) { return sp.s < val; }
        );

        if (it == sampled_.begin())
            return 0;
        if (it == sampled_.end())
            return static_cast<int>(sampled_.size()) - 1;

        int i = std::distance(sampled_.begin(), it);
        int i_prev = i - 1;

        if (std::abs(sampled_[i].s - s_q) < std::abs(sampled_[i_prev].s - s_q))
            return i;
        return i_prev;
    }
    inline int get_sampled_idx_by_traj_pieces_idx(int traj_pieces_idx) const noexcept {
        return std::clamp(traj_pieces_idx + 1, 0, static_cast<int>(sampled_.size()) - 2);
    }
    inline int get_raw_idx_by_traj_pieces_idx(int traj_pieces_idx) const noexcept {
        return sampled_[get_sampled_idx_by_traj_pieces_idx(traj_pieces_idx)].raw_idx0;
    }

    inline static std::vector<double> compute_arc_lengths(const std::vector<Eigen::Vector2d>& path
    ) noexcept {
        std::vector<double> s(path.size(), 0.0);
        for (size_t i = 1; i < path.size(); ++i)
            s[i] = s[i - 1] + (path[i] - path[i - 1]).norm();
        return s;
    }

    inline static SampledPoint interpolate_with_map(
        const std::vector<Eigen::Vector2d>& path,
        const std::vector<double>& s,
        double s_q
    ) noexcept {
        if (s_q <= s.front()) {
            return { path.front(), 0, 0, 0.0, s.front() };
        }

        if (s_q >= s.back()) {
            int n = static_cast<int>(path.size()) - 1;
            return { path.back(), n, n, 0.0, s.back() };
        }

        auto it = std::lower_bound(s.begin(), s.end(), s_q);
        int i = std::distance(s.begin(), it);
        i = std::clamp(i, 1, static_cast<int>(s.size()) - 1);

        double ds = s[i] - s[i - 1];

        if (ds < 1e-6) {
            return { path[i], i, i, 0.0, s_q };
        }

        double t = (s_q - s[i - 1]) / ds;
        Eigen::Vector2d p = path[i - 1] * (1.0 - t) + path[i] * t;

        return { p, i - 1, i, t, s_q };
    }
    inline std::vector<Eigen::Vector2d> get_sampled_pos_vec() const {
        std::vector<Eigen::Vector2d> out;
        out.reserve(sampled_.size());
        for (const auto& sp: sampled_)
            out.push_back(sp.p);
        return out;
    }

    std::vector<Eigen::Vector2d> raw_path_;
    std::vector<double> raw_s_; // raw 的弧长
    // raw_idx -> sampled indices
    std::vector<std::vector<int>> raw_to_sampled_;

    std::vector<SampledPoint> sampled_;

    std::vector<Piece<5, 2>> traj_pieces; //size == sampled_.size() -1
};

} // namespace rose_nav