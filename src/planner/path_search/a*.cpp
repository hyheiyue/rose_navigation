#include "a*.hpp"
#include <memory>
#include <rclcpp/logging.hpp>
namespace rose_nav::planner {
struct AStar::Impl {
    struct Params {
        double safe_radius;
        double clearance_weight;
        double heuristic_weight;
        double obstacle_penalty_weight;
        void load(const ParamsNode& config) {
            safe_radius = config.declare<double>("safe_radius");
            clearance_weight = config.declare<double>("clearance_weight");
            heuristic_weight = config.declare<double>("heuristic_weight");
            obstacle_penalty_weight = config.declare<double>("obstacle_penalty_weight");
        }
    } params_;
    Impl(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
        rose_map_ = rose_map;
        params_.load(config);
        nodes_.reserve(4096);
        succ_buffer_.reserve(8);
    }
    bool is_occupied(const map::VoxelKey<2>& k) const noexcept {
        auto esdf = rose_map_->esdf();
        int idx = esdf->key_to_index(k);
        if (idx < 0)
            return false;
        return esdf->get_esdf(idx) < params_.safe_radius;
    }
    bool check_start_goal_safe(map::VoxelKey<2>& start, map::VoxelKey<2>& goal) const noexcept {
        if (is_occupied(start)) {
            if (!find_nearest_safe(start, start))
                return false;
        }
        if (is_occupied(goal)) {
            if (!find_nearest_safe(goal, goal))
                return false;
        }

        return true;
    }
    bool find_nearest_safe(const map::VoxelKey<2>& seed, map::VoxelKey<2>& out) const noexcept {
        std::queue<map::VoxelKey<2>> q;
        auto hash = [&](const map::VoxelKey<2>& k) -> uint64_t {
            return (uint64_t(uint32_t(k.x())) << 32) | uint32_t(k.y());
        };
        std::unordered_set<uint64_t> vis;

        q.push(seed);
        vis.insert(hash(seed));
        int limit = 500000;

        static const int dx[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
        static const int dy[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

        while (!q.empty() && limit--) {
            auto c = q.front();
            q.pop();
            if (!is_occupied(c)) {
                out = c;
                return true;
            }
            for (int i = 0; i < 8; i++) {
                map::VoxelKey<2> nb { c.x() + dx[i], c.y() + dy[i] };
                auto h = hash(nb);
                if (!vis.count(h)) {
                    vis.insert(h);
                    q.push(nb);
                }
            }
        }
        return false;
    }
    bool project_point_to_map_boundary(
        const Eigen::Vector2f& start_w,
        const Eigen::Vector2f& goal_w,
        Eigen::Vector2f& projected_goal_w
    ) const noexcept {
        const auto& esdf = rose_map_->esdf();
        Eigen::Vector2f dir = goal_w - start_w;
        float len = dir.norm();
        if (len < 1e-3f)
            return false;

        dir /= len;
        float step = 0.5f * esdf->esdf_->voxel_size;
        float t = 0.0f;

        Eigen::Vector2f last_inside = start_w;

        while (t <= len) {
            Eigen::Vector2f p = start_w + dir * t;
            auto key = esdf->world_to_key(p);
            int idx = esdf->key_to_index(key);
            if (idx < 0) {
                projected_goal_w = last_inside;
                return true;
            }
            last_inside = p;
            t += step;
        }

        projected_goal_w = goal_w;
        return true;
    }
    inline float
    heuristic_cached(float esdf_val, const map::VoxelKey<2>& a, const map::VoxelKey<2>& b)
        const noexcept {
        float dx = float(a.x() - b.x());
        float dy = float(a.y() - b.y());
        float dist = std::sqrt(dx * dx + dy * dy);
        float clearance = params_.clearance_weight * (1.0f / (esdf_val + 0.1f));
        return dist + clearance;
    }

    inline void
    getSuccessors(const map::VoxelKey<2>& cur, std::vector<map::VoxelKey<2>>& succ) const noexcept {
        succ.clear();
        static const int dx[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
        static const int dy[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
        for (int i = 0; i < 8; i++) {
            map::VoxelKey<2> nb { cur.x() + dx[i], cur.y() + dy[i] };
            if (!is_occupied(nb))
                succ.push_back(nb);
        }
    }

    SearchState
    search(const Eigen::Vector2d& _start_w, const Eigen::Vector2d& _goal_w, Path& path) noexcept {
        const auto& esdf = rose_map_->esdf();
        auto start_w = _start_w.cast<float>();
        auto goal_w = _goal_w.cast<float>();
        auto start = esdf->world_to_key(start_w);
        auto goal = esdf->world_to_key(goal_w);
        if (!check_start_goal_safe(start, goal)) {
            RCLCPP_WARN(
                rclcpp::get_logger("rose_nav:planner"),
                "no safe start: (%.2f, %.2f),goal: (%.2f, %.2f) found",
                start_w.x(),
                start_w.y(),
                goal_w.x(),
                goal_w.y()
            );
            return SearchState::NO_PATH;
        }
        const int esdf_size = static_cast<int>(esdf->esdf_->grid_size());
        if (esdf_size <= 0) {
            return SearchState::NO_PATH;
        }

        std::vector<int> index_map(esdf_size, -1);

        std::priority_queue<PQItem, std::vector<PQItem>, PQComp> open;

        int start_idx = esdf->key_to_index(start);
        int goal_idx = esdf->key_to_index(goal);
        if (start_idx < 0) {
            RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "start out of map");
            return SearchState::NO_PATH;
        }
        bool goal_projected = false;
        Eigen::Vector2f projected_goal_w;
        if (goal_idx < 0) {
            if (project_point_to_map_boundary(start_w, goal_w, projected_goal_w)) {
                goal_projected = true;
                goal = esdf->world_to_key(projected_goal_w);
            }
        }
        Node sn;
        sn.key = start;
        sn.g = 0.0f;
        sn.esdf = esdf->get_esdf(start_idx);
        float h_start = heuristic_cached(sn.esdf, start, goal);
        sn.f = sn.g + h_start;
        sn.parent = -1;
        nodes_.clear();
        nodes_.push_back(sn);
        index_map[start_idx] = 0;
        open.push({ 0, sn.f, h_start });

        const float heu_weight = params_.heuristic_weight;
        constexpr int max_iters = 10000000;
        int iters = 0;
        auto t0 = std::chrono::steady_clock::now();
        constexpr float max_time = 1.0f;
        constexpr int time_check_interval = 256;

        while (!open.empty()) {
            iters++;
            if ((iters & (time_check_interval - 1)) == 0) {
                float dt =
                    std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
                if (dt > max_time || iters > max_iters) {
                    RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "search timeout");
                    return SearchState::TIMEOUT;
                }
            }

            PQItem top = open.top();
            open.pop();
            int cid = top.id;
            if (cid < 0 || cid >= (int)nodes_.size())
                continue;
            if (top.f > nodes_[cid].f + 1e-4f)
                continue;

            // goal reached
            if (nodes_[cid].key == goal) {
                path.clear();

                for (int id = cid; id >= 0; id = nodes_[id].parent) {
                    auto w = esdf->key_to_world(nodes_[id].key);
                    path.emplace_back(w.x(), w.y());
                }

                std::reverse(path.begin(), path.end());
                if (goal_projected) {
                    path.emplace_back(goal_w.x(), goal_w.y());
                    std::cout << "[A*] append real goal: " << goal_w.transpose() << std::endl;
                }
                return SearchState::SUCCESS;
            }

            getSuccessors(nodes_[cid].key, succ_buffer_);

            for (const auto& nbk: succ_buffer_) {
                int nb_idx = esdf->key_to_index(nbk);
                if (nb_idx < 0 || nb_idx >= esdf_size)
                    continue;

                float base = (std::abs(nodes_[cid].key.x() - nbk.x())
                                  + std::abs(nodes_[cid].key.y() - nbk.y())
                              == 1)
                    ? 1.0f
                    : 1.41421356f;
                float d = esdf->get_esdf(nb_idx);
                float step_cost = base + params_.obstacle_penalty_weight * (1.0f / (d + 0.1f));

                float ng = nodes_[cid].g + step_cost;
                float nh = heuristic_cached(d, nbk, goal) * heu_weight;
                float nf = ng + nh;

                int exist_id = index_map[nb_idx];
                if (exist_id != -1) {
                    if (ng >= nodes_[exist_id].g - 1e-4f)
                        continue;
                    nodes_[exist_id].g = ng;
                    nodes_[exist_id].f = nf;
                    nodes_[exist_id].parent = cid;
                    open.push({ exist_id, nf, nh });
                } else {
                    Node nn;
                    nn.key = nbk;
                    nn.g = ng;
                    nn.f = nf;
                    nn.parent = cid;
                    nn.esdf = d;
                    int nid = (int)nodes_.size();
                    nodes_.push_back(nn);
                    index_map[nb_idx] = nid;
                    open.push({ nid, nf, nh });
                }
            }
        }
        RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "fuck");
        return SearchState::NO_PATH;
    }
    struct Node {
        map::VoxelKey<2> key;
        float g = 0.0f;
        float f = 0.0f;
        int parent = -1;
        float esdf = 0.0f; // cached clearance value
    };
    struct PQItem {
        int id;
        float f;
        float h;
    };

    struct PQComp {
        bool operator()(const PQItem& a, const PQItem& b) const {
            constexpr float EPS = 1e-4f;
            if (std::abs(a.f - b.f) < EPS) {
                return a.h > b.h;
            }
            return a.f > b.f;
        }
    };
    std::vector<Node> nodes_;
    map::RoseMap::Ptr rose_map_;
    mutable std::vector<map::VoxelKey<2>> succ_buffer_;
};
AStar::AStar(map::RoseMap::Ptr rose_map, const ParamsNode& config) {
    _impl = std::make_unique<Impl>(rose_map, config);
}
AStar::~AStar() {
    _impl.reset();
}

AStar::SearchState
AStar::search(const Eigen::Vector2d& _start_w, const Eigen::Vector2d& _goal_w, Path& path) {
    return _impl->search(_start_w, _goal_w, path);
}
} // namespace rose_nav::planner