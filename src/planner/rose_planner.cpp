#include "rose_planner.hpp"
#include "common.hpp"
#include "control/lmpc.hpp"
#include "map/rose_map.hpp"
#include "path_search/a*.hpp"
#include "planner/traj.hpp"
#include "traj_opt/trajectory_opt.hpp"
#include "utils/rcl_tf.hpp"
#include "utils/rclcpp_parameter_node.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <algorithm>
#include <memory>
#include <nav_msgs/msg/path.hpp>
#include <numeric>
#include <optional>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <std_msgs/msg/float64.hpp>
#include <string>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>
namespace rose_nav::planner {
struct RosePlanner::Impl {
    static constexpr bool USE_OPT = true;
    enum FSMSTATE : int {
        INIT,
        WAIT_GOAL,
        SEARCH_PATH,
        REPLAN,
    };
    struct Params {
        double robot_radius;
        std::string target_frame;
        bool use_control_output;
        double default_wz;
        double sample_ds;
        double expected_speed;
        double check_safe_horizon;
        void load(const ParamsNode& config) {
            robot_radius = config.declare<double>("robot_radius");
            target_frame = config.declare<std::string>("target_frame");
            use_control_output = config.declare<bool>("use_control_output");
            default_wz = config.declare<double>("default_wz");
            sample_ds = config.declare<double>("sample_ds");
            expected_speed = config.declare<double>("expected_speed");
            check_safe_horizon = config.declare<double>("check_safe_horizon");
        }
    } params_;
    Impl(rclcpp::Node& node) {
        node_ = &node;
        fsm_ = INIT;
        auto config = ParamsNode(node, "rose_planner");
        params_.load(config);
        tf_ = std::make_unique<RclTF>(node);
        robo_ = Robo::create(node);
        rose_map_ = map::RoseMap::create(node);
        a_star_ = AStar::create(rose_map_, config.sub("a_star"));
        traj_opt_ = TrajOpt::create(rose_map_, config.sub("traj_opt"));
        mpc_ = LMPC::create(rose_map_, config.sub("mpc"));
        auto odom_topic = config.declare<std::string>("odometry_topic");

        odometry_sub_ = node.create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            rclcpp::SensorDataQoS(),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                const auto& odom_in = *msg;

                static Eigen::Isometry3d T;
                if (auto opt = tf_->get_transform(
                        params_.target_frame,
                        odom_in.header.frame_id,
                        odom_in.header.stamp,
                        rclcpp::Duration::from_seconds(0.1)
                    ))
                {
                    T = *opt;
                } else {
                }
                Eigen::Vector3d p(
                    odom_in.pose.pose.position.x,
                    odom_in.pose.pose.position.y,
                    odom_in.pose.pose.position.z
                );
                p = T * p;
                Eigen::Quaterniond q_in(
                    odom_in.pose.pose.orientation.w,
                    odom_in.pose.pose.orientation.x,
                    odom_in.pose.pose.orientation.y,
                    odom_in.pose.pose.orientation.z
                );
                Eigen::Quaterniond q_out(T.rotation() * q_in);
                q_out.normalize();
                geometry_msgs::msg::PoseStamped pose_out;
                pose_out.header = odom_in.header;
                pose_out.header.frame_id = params_.target_frame;
                pose_out.pose.position.x = p.x();
                pose_out.pose.position.y = p.y();
                pose_out.pose.position.z = p.z();
                pose_out.pose.orientation.x = q_out.x();
                pose_out.pose.orientation.y = q_out.y();
                pose_out.pose.orientation.z = q_out.z();
                pose_out.pose.orientation.w = q_out.w();
                Eigen::Vector3d vlin(
                    odom_in.twist.twist.linear.x,
                    odom_in.twist.twist.linear.y,
                    odom_in.twist.twist.linear.z
                );
                Eigen::Vector3d vang(
                    odom_in.twist.twist.angular.x,
                    odom_in.twist.twist.angular.y,
                    odom_in.twist.twist.angular.z
                );
                Eigen::Matrix3d R = T.rotation();
                vlin = R * vlin;
                vang = R * vang;
                nav_msgs::msg::Odometry odom_out = odom_in;
                odom_out.header.frame_id = params_.target_frame;
                odom_out.pose.pose = pose_out.pose;
                odom_out.twist.twist.linear.x = vlin.x();
                odom_out.twist.twist.linear.y = vlin.y();
                odom_out.twist.twist.linear.z = vlin.z();

                odom_out.twist.twist.angular.x = vang.x();
                odom_out.twist.twist.angular.y = vang.y();
                odom_out.twist.twist.angular.z = vang.z();
                robo_->set_current_odom(odom_out);
            }
        );
        std::string goal_topic = config.declare<std::string>("goal_topic");
        goal_sub_ = node.create_subscription<geometry_msgs::msg::PoseStamped>(
            goal_topic,
            10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                static Eigen::Isometry3d target_2_msg = Eigen::Isometry3d::Identity();
                auto target_2_msg_opt = tf_->get_transform(
                    params_.target_frame,
                    msg->header.frame_id,
                    msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1)
                );
                if (target_2_msg_opt) {
                    target_2_msg = *target_2_msg_opt;
                }
                Eigen::Vector4d
                    p(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z, 1.0f);
                p = target_2_msg * p;
                set_goal(p.head<2>());
            }
        );

        goal_point_sub_ = node.create_subscription<geometry_msgs::msg::PointStamped>(
            goal_topic,
            10,
            [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
                static Eigen::Isometry3d target_2_msg = Eigen::Isometry3d::Identity();
                auto target_2_msg_opt = tf_->get_transform(
                    params_.target_frame,
                    msg->header.frame_id,
                    msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1)
                );
                if (target_2_msg_opt) {
                    target_2_msg = *target_2_msg_opt;
                }
                Eigen::Vector4d p(msg->point.x, msg->point.y, msg->point.z, 1.0f);
                p = target_2_msg * p;
                set_goal(p.head<2>());
            }
        );

        predict_path_pub_ = node.create_publisher<nav_msgs::msg::Path>("predict_path", 10);
        cmd_vel_pub_ = node.create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        cmd_vel_norm_pub_ = node.create_publisher<std_msgs::msg::Float64>("/cmd_vel_norm", 10);
        vel_marker_pub_ = node.create_publisher<visualization_msgs::msg::Marker>("/vel_marker", 10);
        now_state_marker_pub_ =
            node.create_publisher<visualization_msgs::msg::Marker>("/now_state_marker", 10);

        raw_path_pub_ = node.create_publisher<nav_msgs::msg::Path>("raw_path", 10);
        opt_path_pub_ = node.create_publisher<nav_msgs::msg::Path>("opt_path", 10);
        opt_marker_pub_ =
            node.create_publisher<visualization_msgs::msg::Marker>("/opt_traj_marker", 10);
        auto plan_fps = config.declare<double>("plan_fps");
        plan_thread_ = std::thread([this, plan_fps]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / plan_fps)
                );
                plan_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
        control_fps_ = config.sub("mpc").declare<double>("control_fps");
        control_thread_ = std::thread([this]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / control_fps_)
                );
                control_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
    }

    void set_goal(const Eigen::Vector2d& p) {
        RCLCPP_INFO(rclcpp::get_logger("rose_nav:planner"), "set goal : %.2f , %.2f", p.x(), p.y());
        switch (goal_mode_) {
            case GoalMode::SINGALE: {
                goal_buf_.pos.push_back(p);
                goal_ = goal_buf_;
                fsm_ = FSMSTATE::SEARCH_PATH;
                goal_buf_.pos.clear();
                break;
            }

            case GoalMode::MULTI: {
                goal_buf_.pos.push_back(p);
                break;
            }
        }
    }
    void control_callback() {
        auto now_state = robo_->get_now_state();
        visualization_msgs::msg::Marker now_state_marker;
        now_state_marker.header.frame_id = params_.target_frame;
        now_state_marker.header.stamp = node_->get_clock()->now();
        now_state_marker.ns = "position";
        now_state_marker.type = visualization_msgs::msg::Marker::SPHERE;
        now_state_marker.scale.x = now_state_marker.scale.y = now_state_marker.scale.z = 0.3;
        now_state_marker.color.a = 1.0;
        now_state_marker.color.g = 1.0;
        now_state_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        now_state_marker.action = visualization_msgs::msg::Marker::ADD;
        now_state_marker.id = 1;
        now_state_marker.pose.position.x = now_state.pos.x();
        now_state_marker.pose.position.y = now_state.pos.y();
        now_state_marker.pose.position.z = 0.0;
        now_state_marker_pub_->publish(now_state_marker);
        if (!has_traj_) {
            if ((fsm_ == FSMSTATE::SEARCH_PATH || fsm_ == FSMSTATE::REPLAN)
                && params_.use_control_output) {
                geometry_msgs::msg::Twist cmd;
                cmd.linear.x = 0.0;
                cmd.linear.y = 0.0;
                cmd.angular.z = 0.0;
                cmd_vel_pub_->publish(cmd);
            }
            return;
        }
        static auto last_fsm = FSMSTATE::INIT;
        if (fsm_ != FSMSTATE::REPLAN) {
            static rclcpp::Time replan_start_time;
            static bool timer_active = false;

            auto now = node_->get_clock()->now();
            if (last_fsm == FSMSTATE::REPLAN) {
                replan_start_time = now;
                timer_active = true;
            }

            if (timer_active) {
                if ((now - replan_start_time).seconds() < 1.0) {
                    if (params_.use_control_output) {
                        geometry_msgs::msg::Twist cmd;
                        cmd.linear.x = 0.0;
                        cmd.linear.y = 0.0;
                        cmd.angular.z = 0.0;
                        cmd_vel_pub_->publish(cmd);
                    }

                } else {
                    timer_active = false;
                }
            }
            last_fsm = fsm_;
            return;
        }
        last_fsm = fsm_;
        mpc_->set_traj(current_traj_);
        mpc_->set_current(now_state);
        auto output_opt = mpc_->solve(std::chrono::duration<double>(1.0 / control_fps_));
        if (output_opt) {
            auto output = output_opt.value();
            if (params_.use_control_output) {
                double yaw = now_state.yaw;
                double vx_world = output.vel.x();
                double vy_world = output.vel.y();
                double vx_body = std::cos(yaw) * vx_world + std::sin(yaw) * vy_world;
                double vy_body = -std::sin(yaw) * vx_world + std::cos(yaw) * vy_world;
                geometry_msgs::msg::Twist cmd;
                cmd.linear.x = vx_body;
                cmd.linear.y = vy_body;

                cmd.angular.z = params_.default_wz;
                cmd_vel_pub_->publish(cmd);
                std_msgs::msg::Float64 cmd_norm;
                cmd_norm.data = std::hypot(vx_world, vy_world);
                cmd_vel_norm_pub_->publish(cmd_norm);
            }
            visualization_msgs::msg::Marker vel_marker;

            vel_marker.header.frame_id = params_.target_frame;
            vel_marker.header.stamp = node_->get_clock()->now();
            output.fill_velocity_arrow(vel_marker, now_state);
            vel_marker_pub_->publish(vel_marker);
            nav_msgs::msg::Path predict_path;
            predict_path.header.frame_id = params_.target_frame;
            predict_path.header.stamp = node_->get_clock()->now();
            output.fill_path(predict_path, now_state);
            predict_path_pub_->publish(predict_path);
        }
    }
    void plan_callback() {
        auto logger = rclcpp::get_logger("rose_nav:planner");

        auto reached = [&](double dist_to_goal) {
            RCLCPP_INFO(logger, "Goal reached (dist=%.2f m), go to next goal", dist_to_goal);
            goal_.pos.pop_front();
            if (goal_.pos.empty()) {
                fsm_ = FSMSTATE::WAIT_GOAL;
                RCLCPP_INFO(logger, "all goal have been reached");
            } else {
                fsm_ = FSMSTATE::SEARCH_PATH;
            }
        };

        auto build_new_path = [&](const Eigen::Vector2d& current_pos,
                                  int cut_raw_idx) -> std::vector<Eigen::Vector2d> {
            const auto& old_raw = current_traj_.raw_path_;
            cut_raw_idx = std::clamp(cut_raw_idx, 0, static_cast<int>(old_raw.size()) - 2);

            std::vector<Eigen::Vector2d> front_path =
                search_once(current_pos, old_raw[cut_raw_idx]);

            std::vector<Eigen::Vector2d> new_raw_path = std::move(front_path);
            new_raw_path.insert(new_raw_path.end(), old_raw.begin() + cut_raw_idx, old_raw.end());
            return new_raw_path;
        };

        auto rebuild_traj = [&](const std::vector<Eigen::Vector2d>& new_raw_path,
                                const RoboState& current,
                                std::optional<std::pair<int, int>> no_opt_range =
                                    std::nullopt) -> bool {
            if (new_raw_path.empty()) {
                return false;
            }

            pub_raw_path(new_raw_path);

            current_traj_.set_raw_path(new_raw_path);
            current_traj_.resample(params_.sample_ds);

            std::vector<Piece<5, 2>> pieces;

            pieces = traj_opt_->optimize(
                current_traj_.sampled_,
                params_.sample_ds / params_.expected_speed,
                current,
                no_opt_range
            );

            if (pieces.empty()) {
                return false;
            }

            current_traj_.traj_pieces = std::move(pieces);
            pub_traj(current_traj_);
            has_traj_ = true;
            return true;
        };
        static int last_how_to_replan = -1;
        constexpr int repaln_by_raw = 1;
        constexpr int repaln_by_traj = 2;
        switch (fsm_) {
            case FSMSTATE::INIT:
            case FSMSTATE::WAIT_GOAL:
                break;

            case FSMSTATE::SEARCH_PATH: {
                if (goal_.pos.empty()) {
                    fsm_ = FSMSTATE::WAIT_GOAL;
                    break;
                }

                auto current = robo_->get_now_state();
                has_traj_ = false;

                auto raw_path = search_once(current.pos, goal_.pos.front());
                if (rebuild_traj(raw_path, current)) {
                    last_how_to_replan = -1;
                    fsm_ = FSMSTATE::REPLAN;
                    RCLCPP_INFO(logger, "create new traj successfully");
                }
                break;
            }

            case FSMSTATE::REPLAN: {
                if (goal_.pos.empty()) {
                    fsm_ = FSMSTATE::WAIT_GOAL;
                    break;
                }

                auto current = robo_->get_now_state();
                Eigen::Vector2d goal_pos = goal_.pos.front();
                double dist_to_goal = (goal_pos - current.pos).norm();

                if (dist_to_goal < 0.5) {
                    reached(dist_to_goal);
                    break;
                }

                auto t_and_dis = current_traj_.get_traj_time_by_pos(current.pos);
                double now_t_in_traj = t_and_dis.first;

                if (t_and_dis.second > 0.5) {
                    RCLCPP_WARN(logger, "now in traj too far %.2f", t_and_dis.second);
                    fsm_ = FSMSTATE::SEARCH_PATH;
                    break;
                }

                double total_duration = current_traj_.get_traj_total_duration();
                if (total_duration <= 0) {
                    reached(dist_to_goal);
                    break;
                }

                constexpr double sample_dt = 0.05;
                double check_end_time =
                    std::min(now_t_in_traj + params_.check_safe_horizon, total_duration);

                int last_unsafe_raw_idx = -1;
                double last_unsafe_traj_time = -1.0;

                for (double t = now_t_in_traj; t <= check_end_time; t += sample_dt) {
                    int traj_idx = current_traj_.locate_traj_piece_idx_by_time(t).first;
                    int raw_idx = current_traj_.get_raw_idx_by_traj_pieces_idx(traj_idx);

                    if (!is_safe(current_traj_.raw_path_[raw_idx])) {
                        last_unsafe_raw_idx = raw_idx;
                    }

                    if (!is_safe(current_traj_.get_traj_pos_by_time(t))) {
                        last_unsafe_traj_time = t;
                    }
                }

                auto get_no_opt_end = [&]() {
                    int no_opt_end = static_cast<int>(current_traj_.sampled_.size()) - 1;
                    for (int i = 0; i < static_cast<int>(current_traj_.sampled_.size()); ++i) {
                        if (current_traj_.sampled_[i].s / params_.expected_speed
                            > params_.check_safe_horizon) {
                            no_opt_end = i;
                            break;
                        }
                    }
                    return no_opt_end;
                };
                if (last_unsafe_raw_idx != -1) {
                    RCLCPP_INFO(
                        logger,
                        "raw path unsafe last_unsafe_raw_idx: %d",
                        last_unsafe_raw_idx
                    );
                    has_traj_ = false;

                    auto new_raw_path = build_new_path(current.pos, last_unsafe_raw_idx);

                    int no_opt_end = get_no_opt_end();
                    bool traj_valid =
                        rebuild_traj(new_raw_path, current, std::make_pair(0, no_opt_end));

                    last_how_to_replan = repaln_by_raw;
                    if (rebuild_traj(new_raw_path, current)) {
                        RCLCPP_INFO(logger, "local replan success");
                    } else {
                        fsm_ = FSMSTATE::SEARCH_PATH;
                    }

                    break;
                } else if (last_unsafe_traj_time >= 0) {
                    last_how_to_replan = repaln_by_traj;
                    RCLCPP_INFO(
                        logger,
                        "traj unsafe last_unsafe_traj_time: %f",
                        last_unsafe_traj_time
                    );
                    has_traj_ = false;

                    double no_opt_end_t =
                        std::min(now_t_in_traj + params_.check_safe_horizon, total_duration);
                    int end_traj_piece_idx =
                        current_traj_.locate_traj_piece_idx_by_time(no_opt_end_t).first;
                    int end_raw_path_idx =
                        current_traj_.get_raw_idx_by_traj_pieces_idx(end_traj_piece_idx);

                    auto new_raw_path = build_new_path(current.pos, end_raw_path_idx);
                    if (new_raw_path.empty()) {
                        fsm_ = FSMSTATE::SEARCH_PATH;
                        break;
                    }

                    int no_opt_end = get_no_opt_end();

                    if (rebuild_traj(new_raw_path, current, std::make_pair(0, no_opt_end))) {
                        RCLCPP_INFO(logger, "local replan success no_opt_end: %d", no_opt_end);
                    } else {
                        fsm_ = FSMSTATE::SEARCH_PATH;
                    }
                    break;
                } else {
                }

                break;
            }
        }
    }
    std::vector<Eigen::Vector2d>
    search_once(const Eigen::Vector2d& start, const Eigen::Vector2d& goal) {
        auto [path, search_state] = search(start, goal);
        if (search_state == AStar::SearchState::NO_PATH) {
            RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "No path found by A*");
            return {};
        } else if (search_state == AStar::SearchState::TIMEOUT) {
            RCLCPP_WARN(rclcpp::get_logger("rose_nav:planner"), "A* search timeout");
            return {};
        }
        return path;
    }
    bool is_safe(const Eigen::Vector2d& p) const noexcept {
        auto esdf = rose_map_->esdf();
        auto key = esdf->world_to_key(p.cast<float>());
        int idx = esdf->key_to_index(key);

        if (idx < 0)
            return true;

        if (esdf->get_esdf(idx) < params_.robot_radius) {
            return false;
        }

        return true;
    };

    std::pair<std::vector<Eigen::Vector2d>, AStar::SearchState>
    search(const Eigen::Vector2d& start, const Eigen::Vector2d& goal) {
        std::vector<Eigen::Vector2d> path;
        AStar::SearchState search_state = AStar::SearchState::NO_PATH;

        try {
            search_state = a_star_->search(start, goal, path);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(
                rclcpp::get_logger("rose_nav:planner"),
                "A* search exception: %s",
                e.what()
            );
        }

        return std::make_pair(path, search_state);
    }
    void pub_raw_path(const std::vector<Eigen::Vector2d>& path) {
        nav_msgs::msg::Path raw_path_msg;
        raw_path_msg.header.stamp = node_->now();
        raw_path_msg.header.frame_id = params_.target_frame;
        if (raw_path_pub_->get_subscription_count() > 0) {
            for (int i = 0; i < (int)path.size(); i++) {
                geometry_msgs::msg::PoseStamped pose_msg;
                pose_msg.header = raw_path_msg.header;
                pose_msg.pose.position.x = path[i].x();
                pose_msg.pose.position.y = path[i].y();
                pose_msg.pose.position.z = 0.0;

                raw_path_msg.poses.push_back(pose_msg);
            }
            raw_path_pub_->publish(raw_path_msg);
        }
    }
    void pub_traj(const Traj& traj) const {
        if ((opt_path_pub_->get_subscription_count() > 0
             || opt_marker_pub_->get_subscription_count() > 0)
            && traj.traj_pieces.size() > 2)
        {
            double dt = 0.05;
            nav_msgs::msg::Path opt_path_msg;
            opt_path_msg.header.stamp = node_->now();
            opt_path_msg.header.frame_id = params_.target_frame;
            double t_cur = 0.0;
            opt_path_msg.poses.clear();

            visualization_msgs::msg::Marker marker;
            marker.header = opt_path_msg.header;
            marker.ns = "opt_traj";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            marker.color.r = 0.1f;
            marker.color.g = 0.1f;
            marker.color.b = 0.9f;
            marker.color.a = 1.0f;

            marker.points.clear();
            double total_duration = traj.get_traj_total_duration();
            while (t_cur <= total_duration) {
                Eigen::Vector2d pos = traj.get_traj_pos_by_time(t_cur);
                Eigen::Vector2d vel = traj.get_traj_vel_by_time(t_cur);
                double yaw =
                    std::hypot(vel.x(), vel.y()) > 1e-3 ? std::atan2(vel.y(), vel.x()) : 0.0;
                geometry_msgs::msg::PoseStamped p;
                p.header = opt_path_msg.header;
                p.pose.position.x = pos.x();
                p.pose.position.y = pos.y();
                p.pose.position.z = 0.0;

                tf2::Quaternion q;
                q.setRPY(0, 0, yaw);
                q.normalize();
                p.pose.orientation = tf2::toMsg(q);

                opt_path_msg.poses.push_back(p);
                geometry_msgs::msg::Point mp;
                mp.x = pos.x();
                mp.y = pos.y();
                mp.z = p.pose.position.z;
                marker.points.push_back(mp);
                t_cur += dt;
            }
            opt_path_pub_->publish(opt_path_msg);
            opt_marker_pub_->publish(marker);
        }
    }

    rclcpp::Node* node_;

    FSMSTATE fsm_;
    Robo::Ptr robo_;
    AStar::Ptr a_star_;
    map::RoseMap::Ptr rose_map_;
    TrajOpt::Ptr traj_opt_;
    LMPC::Ptr mpc_;
    RclTF::Ptr tf_;
    Goal goal_buf_;
    Goal goal_;

    enum class GoalMode : u_int8_t {
        SINGALE = 0,
        MULTI = 1,
    } goal_mode_ = GoalMode::SINGALE;

    std::thread plan_thread_;
    std::thread control_thread_;
    double control_fps_;
    Traj current_traj_;
    bool has_traj_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_point_sub_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr opt_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr opt_marker_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predict_path_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_vel_norm_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vel_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr now_state_marker_pub_;
};
RosePlanner::RosePlanner(rclcpp::Node& node) {
    _impl = std::make_unique<Impl>(node);
}
RosePlanner::~RosePlanner() {
    _impl.reset();
}
} // namespace rose_nav::planner