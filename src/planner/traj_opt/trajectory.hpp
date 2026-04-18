/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#pragma once

#include "root_finder.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

template<int D, int Freedom>
class Piece {
public:
    typedef Eigen::Matrix<double, Freedom, D + 1> CoefficientMat;
    typedef Eigen::Matrix<double, Freedom, D> VelCoefficientMat;
    typedef Eigen::Matrix<double, Freedom, D - 1> AccCoefficientMat;

private:
    double duration;
    CoefficientMat coeffMat;

public:
    Piece() = default;

    Piece(double dur, const CoefficientMat& cMat): duration(dur), coeffMat(cMat) {}

    inline int getDim() const {
        return Freedom;
    }

    inline int getDegree() const {
        return D;
    }

    inline double getDuration() const {
        return duration;
    }

    inline const CoefficientMat& getCoeffMat() const {
        return coeffMat;
    }

    inline Eigen::VectorXd getPos(const double& t) const {
        Eigen::VectorXd pos;
        pos.resize(Freedom);
        pos.setZero();
        double tn = 1.0;
        for (int i = D; i >= 0; i--) {
            pos += tn * coeffMat.col(i);
            tn *= t;
        }
        return pos;
    }
    inline double getYaw(const double& t) const {
        // 用速度方向计算 yaw
        Eigen::VectorXd v = getVel(t);
        if (v.size() < 2)
            return 0.0;
        double vx = v(0);
        double vy = v(1);
        // 速度过小时方向不可信，保护一下
        if (std::hypot(vx, vy) < 1e-6) {
            return getYaw(0.0); // 回退到起点方向
        }
        return std::atan2(vy, vx);
    }

    inline double getYawDot(const double& t) const {
        // yaw 导数公式 (ax*vy - ay*vx)/(vx^2+vy^2)
        Eigen::VectorXd v = getVel(t);
        Eigen::VectorXd a = getAcc(t);
        if (v.size() < 2 || a.size() < 2)
            return 0.0;

        double vx = v(0);
        double vy = v(1);
        double ax = a(0);
        double ay = a(1);

        double v2 = vx * vx + vy * vy;
        if (v2 < 1e-12)
            return 0.0; // 速度太小，角速度设 0 避免爆炸
        return (ax * vy - ay * vx) / v2;
    }

    inline Eigen::VectorXd getVel(const double& t) const {
        Eigen::VectorXd vel;
        vel.resize(Freedom);
        vel.setZero();
        double tn = 1.0;
        int n = 1;
        for (int i = D - 1; i >= 0; i--) {
            vel += n * tn * coeffMat.col(i);
            tn *= t;
            n++;
        }
        return vel;
    }

    inline Eigen::VectorXd getAcc(const double& t) const {
        Eigen::VectorXd acc;
        acc.resize(Freedom);
        acc.setZero();
        double tn = 1.0;
        int m = 1;
        int n = 2;
        for (int i = D - 2; i >= 0; i--) {
            acc += m * n * tn * coeffMat.col(i);
            tn *= t;
            m++;
            n++;
        }
        return acc;
    }

    inline Eigen::VectorXd getJer(const double& t) const {
        Eigen::VectorXd jer;
        jer.resize(Freedom);
        jer.setZero();
        double tn = 1.0;
        int l = 1;
        int m = 2;
        int n = 3;
        for (int i = D - 3; i >= 0; i--) {
            jer += l * m * n * tn * coeffMat.col(i);
            tn *= t;
            l++;
            m++;
            n++;
        }
        return jer;
    }

    inline Eigen::VectorXd getSna(const double& t) const {
        Eigen::VectorXd sna;
        sna.resize(Freedom);
        sna.setZero();
        double tn = 1.0;
        int l = 1;
        int m = 2;
        int n = 3;
        int o = 4;
        for (int i = D - 4; i >= 0; i--) {
            sna += l * m * n * o * tn * coeffMat.col(i);
            tn *= t;
            l++;
            m++;
            n++;
            o++;
        }
        return sna;
    }

    inline CoefficientMat normalizePosCoeffMat() const {
        CoefficientMat nPosCoeffsMat;
        double t = 1.0;
        for (int i = D; i >= 0; i--) {
            nPosCoeffsMat.col(i) = coeffMat.col(i) * t;
            t *= duration;
        }
        return nPosCoeffsMat;
    }

    inline VelCoefficientMat normalizeVelCoeffMat() const {
        VelCoefficientMat nVelCoeffMat;
        int n = 1;
        double t = duration;
        for (int i = D - 1; i >= 0; i--) {
            nVelCoeffMat.col(i) = n * coeffMat.col(i) * t;
            t *= duration;
            n++;
        }
        return nVelCoeffMat;
    }

    inline AccCoefficientMat normalizeAccCoeffMat() const {
        AccCoefficientMat nAccCoeffMat;
        int n = 2;
        int m = 1;
        double t = duration * duration;
        for (int i = D - 2; i >= 0; i--) {
            nAccCoeffMat.col(i) = n * m * coeffMat.col(i) * t;
            n++;
            m++;
            t *= duration;
        }
        return nAccCoeffMat;
    }

    inline double getMaxVelRate() const {
        VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
        Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0))
            + RootFinder::polySqr(nVelCoeffMat.row(1)) + RootFinder::polySqr(nVelCoeffMat.row(2));
        int N = coeff.size();
        int n = N - 1;
        for (int i = 0; i < N; i++) {
            coeff(i) *= n;
            n--;
        }
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
            return getVel(0.0).norm();
        } else {
            double l = -0.0625;
            double r = 1.0625;
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
                l = 0.5 * l;
            }
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
                r = 0.5 * (r + 1.0);
            }
            std::set<double> candidates =
                RootFinder::solvePolynomial(coeff.head(N - 1), l, r, FLT_EPSILON / duration);
            candidates.insert(0.0);
            candidates.insert(1.0);
            double maxVelRateSqr = -INFINITY;
            double tempNormSqr;
            for (std::set<double>::const_iterator it = candidates.begin(); it != candidates.end();
                 it++) {
                if (0.0 <= *it && 1.0 >= *it) {
                    tempNormSqr = getVel((*it) * duration).squaredNorm();
                    maxVelRateSqr = maxVelRateSqr < tempNormSqr ? tempNormSqr : maxVelRateSqr;
                }
            }
            return sqrt(maxVelRateSqr);
        }
    }

    inline double getMaxAccRate() const {
        AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
        Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0))
            + RootFinder::polySqr(nAccCoeffMat.row(1)) + RootFinder::polySqr(nAccCoeffMat.row(2));
        int N = coeff.size();
        int n = N - 1;
        for (int i = 0; i < N; i++) {
            coeff(i) *= n;
            n--;
        }
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
            return getAcc(0.0).norm();
        } else {
            double l = -0.0625;
            double r = 1.0625;
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
                l = 0.5 * l;
            }
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
                r = 0.5 * (r + 1.0);
            }
            std::set<double> candidates =
                RootFinder::solvePolynomial(coeff.head(N - 1), l, r, FLT_EPSILON / duration);
            candidates.insert(0.0);
            candidates.insert(1.0);
            double maxAccRateSqr = -INFINITY;
            double tempNormSqr;
            for (std::set<double>::const_iterator it = candidates.begin(); it != candidates.end();
                 it++) {
                if (0.0 <= *it && 1.0 >= *it) {
                    tempNormSqr = getAcc((*it) * duration).squaredNorm();
                    maxAccRateSqr = maxAccRateSqr < tempNormSqr ? tempNormSqr : maxAccRateSqr;
                }
            }
            return sqrt(maxAccRateSqr);
        }
    }

    inline bool checkMaxVelRate(const double& maxVelRate) const {
        double sqrMaxVelRate = maxVelRate * maxVelRate;
        if (getVel(0.0).squaredNorm() >= sqrMaxVelRate
            || getVel(duration).squaredNorm() >= sqrMaxVelRate) {
            return false;
        } else {
            VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0))
                + RootFinder::polySqr(nVelCoeffMat.row(1))
                + RootFinder::polySqr(nVelCoeffMat.row(2));
            double t2 = duration * duration;
            coeff.tail<1>()(0) -= sqrMaxVelRate * t2;
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
        }
    }

    inline bool checkMaxAccRate(const double& maxAccRate) const {
        double sqrMaxAccRate = maxAccRate * maxAccRate;
        if (getAcc(0.0).squaredNorm() >= sqrMaxAccRate
            || getAcc(duration).squaredNorm() >= sqrMaxAccRate) {
            return false;
        } else {
            AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0))
                + RootFinder::polySqr(nAccCoeffMat.row(1))
                + RootFinder::polySqr(nAccCoeffMat.row(2));
            double t2 = duration * duration;
            double t4 = t2 * t2;
            coeff.tail<1>()(0) -= sqrMaxAccRate * t4;
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
        }
    }
    static double binomialCoeff(int n, int k) {
        if (k < 0 || k > n)
            return 0.0;
        if (k == 0 || k == n)
            return 1.0;
        double res = 1.0;
        for (int i = 1; i <= k; ++i) {
            res *= (n - (k - i));
            res /= i;
        }
        return res;
    }

    static CoefficientMat shiftCoeffMat(const CoefficientMat& cMat, double t0) {
        CoefficientMat newMat;
        newMat.setZero();

        for (int i = 0; i <= D; ++i) {
            int n = D - i;
            const auto& ci = cMat.col(i);

            double t0_pow = 1.0;
            for (int j = n; j >= 0; --j) {
                int idx = D - j;
                newMat.col(idx) += ci * binomialCoeff(n, j) * t0_pow;
                t0_pow *= t0;
            }
        }
        return newMat;
    }

    inline Piece shift(double t0) const {
        return Piece(duration - t0, shiftCoeffMat(coeffMat, t0));
    }
};

template<int D, int Freedom>
class Trajectory {
private:
    typedef std::vector<Piece<D, Freedom>> Pieces;
    Pieces pieces;

public:
    Trajectory() = default;

    Trajectory(
        const std::vector<double>& durs,
        const std::vector<typename Piece<D, Freedom>::CoefficientMat>& cMats
    ) {
        int N = std::min(durs.size(), cMats.size());
        pieces.reserve(N);
        for (int i = 0; i < N; i++) {
            pieces.emplace_back(durs[i], cMats[i]);
        }
    }
    inline std::vector<Eigen::Vector2d> toPointVector(double dt) const {
        std::vector<Eigen::Vector2d> pts;
        if (pieces.empty())
            return pts;

        double totalT = getTotalDuration();
        double n_est = 0.0;

        if (dt > 1e-9 && totalT > 0.0 && std::isfinite(dt) && std::isfinite(totalT)) {
            n_est = std::ceil(totalT / dt) + 2;
        }

        size_t N = 0;
        if (n_est > 0.0 && n_est < 1e7) {
            N = static_cast<size_t>(n_est);
        }

        const size_t MAX_RESERVE = 5000;
        N = std::min(N, MAX_RESERVE);

        pts.reserve(N);

        for (int i = 0; i < getPieceNum(); ++i) {
            double T = pieces[i].getDuration();
            double t = 0.0;

            while (t < T) {
                pts.push_back(pieces[i].getPos(t));
                t += dt;
            }

            pts.push_back(pieces[i].getPos(T));
        }

        return pts;
    }

    inline int getPieceNum() const {
        return pieces.size();
    }

    inline int getDim() const {
        return Freedom;
    }
    inline Eigen::VectorXd getDurations() const {
        int N = getPieceNum();
        Eigen::VectorXd durations(N);
        for (int i = 0; i < N; i++) {
            durations(i) = pieces[i].getDuration();
        }
        return durations;
    }

    inline double getTotalDuration() const {
        int N = getPieceNum();
        double totalDuration = 0.0;
        for (int i = 0; i < N; i++) {
            totalDuration += pieces[i].getDuration();
        }
        return totalDuration;
    }

    inline Eigen::MatrixXd getPositions() const {
        int N = getPieceNum();
        Eigen::MatrixXd positions(Freedom, N + 1);
        for (int i = 0; i < N; i++) {
            positions.col(i) = pieces[i].getCoeffMat().col(D);
        }
        positions.col(N) = pieces[N - 1].getPos(pieces[N - 1].getDuration());
        return positions;
    }

    inline const Piece<D, Freedom>& operator[](int i) const {
        return pieces[i];
    }

    inline Piece<D, Freedom>& operator[](int i) {
        return pieces[i];
    }

    inline void clear(void) {
        pieces.clear();
        return;
    }

    inline typename Pieces::const_iterator begin() const {
        return pieces.begin();
    }

    inline typename Pieces::const_iterator end() const {
        return pieces.end();
    }

    inline typename Pieces::iterator begin() {
        return pieces.begin();
    }

    inline typename Pieces::iterator end() {
        return pieces.end();
    }

    inline void reserve(const int& n) {
        pieces.reserve(n);
        return;
    }

    inline void emplace_back(const Piece<D, Freedom>& piece) {
        pieces.emplace_back(piece);
        return;
    }

    inline void
    emplace_back(const double& dur, const typename Piece<D, Freedom>::CoefficientMat& cMat) {
        pieces.emplace_back(dur, cMat);
        return;
    }

    inline void append(const Trajectory<D, Freedom>& traj) {
        pieces.insert(pieces.end(), traj.begin(), traj.end());
        return;
    }

    inline int locatePieceIdx(double& t) const {
        int N = getPieceNum();
        int idx;
        double dur;
        for (idx = 0; idx < N && t > (dur = pieces[idx].getDuration()); idx++) {
            t -= dur;
        }
        if (idx == N) {
            idx--;
            t += pieces[idx].getDuration();
        }
        return idx;
    }

    inline Eigen::VectorXd getPos(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getPos(t);
    }
    inline double getYaw(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getYaw(t);
    }
    inline double getYawDot(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getYawDot(t);
    }
    inline Eigen::VectorXd getVel(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getVel(t);
    }

    inline Eigen::VectorXd getAcc(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAcc(t);
    }

    inline Eigen::VectorXd getJer(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getJer(t);
    }

    inline Eigen::VectorXd getSna(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getSna(t);
    }

    inline Eigen::VectorXd getJuncPos(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D);
        } else {
            return pieces[juncIdx - 1].getPos(pieces[juncIdx - 1].getDuration());
        }
    }

    inline Eigen::VectorXd getJuncVel(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D - 1);
        } else {
            return pieces[juncIdx - 1].getVel(pieces[juncIdx - 1].getDuration());
        }
    }

    inline Eigen::VectorXd getJuncAcc(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D - 2) * 2.0;
        } else {
            return pieces[juncIdx - 1].getAcc(pieces[juncIdx - 1].getDuration());
        }
    }

    inline double getMaxVelRate() const {
        int N = getPieceNum();
        double maxVelRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++) {
            tempNorm = pieces[i].getMaxVelRate();
            maxVelRate = maxVelRate < tempNorm ? tempNorm : maxVelRate;
        }
        return maxVelRate;
    }

    inline double getMaxAccRate() const {
        int N = getPieceNum();
        double maxAccRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++) {
            tempNorm = pieces[i].getMaxAccRate();
            maxAccRate = maxAccRate < tempNorm ? tempNorm : maxAccRate;
        }
        return maxAccRate;
    }

    inline bool checkMaxVelRate(const double& maxVelRate) const {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++) {
            feasible = feasible && pieces[i].checkMaxVelRate(maxVelRate);
        }
        return feasible;
    }

    inline bool checkMaxAccRate(const double& maxAccRate) const {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++) {
            feasible = feasible && pieces[i].checkMaxAccRate(maxAccRate);
        }
        return feasible;
    }
    inline std::pair<double, double>
    getTimeByPos(const Eigen::Vector2d& pos, double smaple_dt = 0.05) const {
        double best_t = 0.0;
        double t = 0.0;
        double total_duration = getTotalDuration();
        double min_dis = std::numeric_limits<double>::infinity();
        while (t <= total_duration) {
            auto p = getPos(t);
            auto dis = (p - pos).norm();
            if (dis < min_dis) {
                min_dis = dis;
                best_t = t;
            }
            t += smaple_dt;
        }
        return std::make_pair(best_t, min_dis);
    }

    inline std::vector<Eigen::Vector2d>
    toPointVectorBySpacing(double ds, double dt_search = 0.01) const {
        std::vector<Eigen::Vector2d> pts;

        if (pieces.empty() || ds <= 1e-6 || dt_search <= 1e-6)
            return pts;

        const double totalT = getTotalDuration();
        if (totalT <= 0.0 || !std::isfinite(totalT))
            return pts;

        pts.reserve(2000);

        double t = 0.0;
        Eigen::Vector2d last_pt = getPos(0.0);
        pts.push_back(last_pt);

        double acc_dist = 0.0;

        while (t < totalT) {
            t += dt_search;
            if (t > totalT)
                t = totalT;

            Eigen::Vector2d cur_pt = getPos(t);
            double d = (cur_pt - last_pt).norm();
            acc_dist += d;

            if (acc_dist >= ds) {
                pts.push_back(cur_pt);
                acc_dist = 0.0;
            }

            last_pt = cur_pt;
        }

        Eigen::Vector2d end_pt = getPos(totalT);
        if (pts.empty() || (pts.back() - end_pt).norm() > 1e-6) {
            pts.push_back(end_pt);
        }

        return pts;
    }
    inline void truncateBeforeTime(double t) {
        if (pieces.empty())
            return;

        if (t <= 0.0)
            return;

        double totalT = getTotalDuration();
        if (t >= totalT) {
            pieces.clear();
            return;
        }

        double local_t = t;
        int idx = locatePieceIdx(local_t);

        Pieces new_pieces;
        new_pieces.reserve(pieces.size() - idx);

        const Piece<D, Freedom>& old_piece = pieces[idx];
        const double old_T = old_piece.getDuration();
        const double new_T = old_T - local_t;

        Eigen::VectorXd p0 = old_piece.getPos(local_t);
        Eigen::VectorXd v0 = old_piece.getVel(local_t);
        Eigen::VectorXd a0 = old_piece.getAcc(local_t);
        typename Piece<D, Freedom>::CoefficientMat new_cMat = old_piece.getCoeffMat();

        new_cMat = Piece<D, Freedom>::shiftCoeffMat(new_cMat, local_t);

        new_pieces.emplace_back(new_T, new_cMat);
        for (int i = idx + 1; i < pieces.size(); ++i) {
            new_pieces.push_back(pieces[i]);
        }

        pieces.swap(new_pieces);
    }
};
