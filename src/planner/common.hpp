#pragma once
#include "angles.h"
#include "planner/traj_opt/trajectory.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <deque>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/node.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
namespace rose_nav::planner {
using TrajType = Trajectory<5, 2>;
struct RoboState {
    Eigen::Vector2d pos;
    Eigen::Vector2d vel;
    double yaw;
};
class Robo {
public:
    using Ptr = std::unique_ptr<Robo>;
    Robo(rclcpp::Node& node) {
        node_ = &node;
    }
    static Ptr create(rclcpp::Node& node) {
        return std::make_unique<Robo>(node);
    }
    void set_current_odom(const nav_msgs::msg::Odometry& current_odom) {
        current_odom_ = current_odom;
    }
    template<class Tag>
    static inline double orientation_to_yaw(const geometry_msgs::msg::Quaternion& q) noexcept {
        // Get armor yaw
        tf2::Quaternion tf_q;
        tf2::fromMsg(q, tf_q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        // Make yaw change continuous (-pi~pi to -inf~inf)
        static double last_yaw_ = 0.0;
        yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
        last_yaw_ = yaw;
        return yaw;
    }
    RoboState get_now_state() const noexcept {
        RoboState now_state;
        auto current = get_now_pose();
        now_state.pos.x() = current.pose.position.x;
        now_state.pos.y() = current.pose.position.y;
        auto vel =
            Eigen::Vector2d(current_odom_.twist.twist.linear.x, current_odom_.twist.twist.linear.y);
        now_state.vel = vel;
        struct Tag {};
        double now_yaw = orientation_to_yaw<Tag>(current.pose.orientation);
        now_state.yaw = now_yaw;
        return now_state;
    }
    geometry_msgs::msg::PoseStamped get_now_pose() const noexcept {
        geometry_msgs::msg::PoseStamped predicted_pose;
        predicted_pose.header.stamp = node_->get_clock()->now();
        predicted_pose.header.frame_id = current_odom_.header.frame_id;

        double px = current_odom_.pose.pose.position.x;
        double py = current_odom_.pose.pose.position.y;
        double pz = current_odom_.pose.pose.position.z;

        Eigen::Quaterniond q(
            current_odom_.pose.pose.orientation.w,
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z
        );

        Eigen::Vector3d v(
            current_odom_.twist.twist.linear.x,
            current_odom_.twist.twist.linear.y,
            current_odom_.twist.twist.linear.z
        );

        Eigen::Vector3d w(
            current_odom_.twist.twist.angular.x,
            current_odom_.twist.twist.angular.y,
            current_odom_.twist.twist.angular.z
        );

        auto now = node_->get_clock()->now();
        auto stamp = rclcpp::Time(current_odom_.header.stamp, now.get_clock_type());

        double dt = (now - stamp).seconds();
        // dt =0.0;
        Eigen::Quaterniond dq = Eigen::Quaterniond(Eigen::AngleAxisd(
            dt * w.norm(),
            (w.norm() > 1e-6) ? w.normalized() : Eigen::Vector3d::UnitZ()
        ));

        Eigen::Quaterniond q_pred = q * dq;
        q_pred.normalize();
        Eigen::Vector3d p_pred(px, py, pz);
        p_pred += v * dt;
        predicted_pose.pose.position.x = p_pred.x();
        predicted_pose.pose.position.y = p_pred.y();
        predicted_pose.pose.position.z = p_pred.z();

        predicted_pose.pose.orientation.w = q_pred.w();
        predicted_pose.pose.orientation.x = q_pred.x();
        predicted_pose.pose.orientation.y = q_pred.y();
        predicted_pose.pose.orientation.z = q_pred.z();

        return predicted_pose;
    }
    rclcpp::Node* node_;
    nav_msgs::msg::Odometry current_odom_;
};
struct Goal {
    std::deque<Eigen::Vector2d> pos;
};
struct TrajPoint {
    Eigen::Vector2d pos;
    Eigen::Vector2d vel;
    Eigen::Vector2d acc;
    double yaw;
    double w;
};
struct ControlOutput {
    Eigen::Vector2d vel;
    double w;
    std::vector<RoboState> pred_states;
    void fill_path(nav_msgs::msg::Path& predict_path, const RoboState& current) const {
        geometry_msgs::msg::PoseStamped pose_msg;
        for (int i = 0; i < pred_states.size(); ++i) {
            pose_msg.header = predict_path.header;
            pose_msg.pose.position.x = pred_states[i].pos.x();
            pose_msg.pose.position.y = pred_states[i].pos.y();
            pose_msg.pose.position.z = 0.0;
            double yaw = std::hypot(pred_states[i].vel.x(), pred_states[i].vel.y()) > 1e-3
                ? std::atan2(pred_states[i].vel.y(), pred_states[i].vel.x())
                : 0.0;
            tf2::Quaternion q;
            q.setRPY(0, 0, yaw);
            q.normalize();
            pose_msg.pose.orientation = tf2::toMsg(q);
            predict_path.poses.push_back(pose_msg);
        }
    }
    void
    fill_velocity_arrow(visualization_msgs::msg::Marker& marker, const RoboState& current) const {
        marker.ns = "mpc_velocity";
        marker.id = 0;

        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;
        auto pos = current.pos.cast<float>();
        geometry_msgs::msg::Point p_start, p_end;
        p_start.x = pos.x();
        p_start.y = pos.y();
        p_start.z = 0.0;
        double scale = 3.0;
        auto vel_normalized = vel.normalized();
        p_end.x = pos.x() + scale * vel_normalized.x();
        p_end.y = pos.y() + scale * vel_normalized.y();
        p_end.z = 0.0;

        marker.points.push_back(p_start);
        marker.points.push_back(p_end);

        marker.scale.x = 0.1; // shaft diameter
        marker.scale.y = 0.2; // head diameter
        marker.scale.z = 0.3; // head length

        marker.color.r = 0.9f;
        marker.color.g = 0.1f;
        marker.color.b = 0.1f;
        marker.color.a = 1.0f;

        marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    }
};
} // namespace rose_nav::planner