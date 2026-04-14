#pragma once
#include "utils.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Geometry/Transform.h>
#include <optional>
#include <tf2/convert.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
namespace rose_nav {
class RclTF {
public:
    using Ptr = std::unique_ptr<RclTF>;
    RclTF(rclcpp::Node& n) {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(n.get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(n);
        node_ = &n;
    }
    Eigen::Isometry3d tf2eigen(const geometry_msgs::msg::TransformStamped& tf) const noexcept {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() << tf.transform.translation.x, tf.transform.translation.y,
            tf.transform.translation.z;
        auto q = tf.transform.rotation;
        Eigen::Quaterniond Q(q.w, q.x, q.y, q.z);
        T.linear() = Q.toRotationMatrix();
        return T;
    }
    std::optional<tf2::Transform>
    get_tf2_transform(const std::string& target, const std::string& source, rclcpp::Time t)
        const noexcept {
        tf2::Transform tf;
        try {
            geometry_msgs::msg::TransformStamped tf_msg =
                tf_buffer_->lookupTransform(target, source, t);

            tf2::fromMsg(tf_msg.transform, tf);
            return tf;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
        return tf;
    }
    std::optional<tf2::Transform> get_tf2_transform(
        const std::string& target,
        const std::string& source,
        rclcpp::Time t,
        const rclcpp::Duration& timeout
    ) const noexcept {
        tf2::Transform tf;
        try {
            geometry_msgs::msg::TransformStamped tf_msg =
                tf_buffer_->lookupTransform(target, source, t, timeout);

            tf2::fromMsg(tf_msg.transform, tf);
            return tf;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
        return tf;
    }
    std::optional<Eigen::Isometry3d>
    get_transform(const std::string& target, const std::string& source, rclcpp::Time t)
        const noexcept {
        Eigen::Isometry3d T_out = Eigen::Isometry3d::Identity();
        try {
            geometry_msgs::msg::TransformStamped tf_msg =
                tf_buffer_->lookupTransform(target, source, t);

            T_out = tf2eigen(tf_msg);
            return T_out;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
        return T_out;
    }
    std::optional<Eigen::Isometry3d> get_transform(
        const std::string& target,
        const std::string& source,
        rclcpp::Time t,
        const rclcpp::Duration& timeout
    ) const noexcept {
        Eigen::Isometry3d T_out = Eigen::Isometry3d::Identity();
        try {
            geometry_msgs::msg::TransformStamped tf_msg =
                tf_buffer_->lookupTransform(target, source, t, timeout);

            T_out = tf2eigen(tf_msg);
            return T_out;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
        return T_out;
    }

    void publish_transform(
        const Eigen::Isometry3d& transform,
        const std::string& parent_frame,
        const std::string& child_frame,
        const rclcpp::Time& stamp
    ) const noexcept {
        geometry_msgs::msg::TransformStamped tmsg;
        tmsg.header.stamp = stamp;
        tmsg.header.frame_id = parent_frame;
        tmsg.child_frame_id = child_frame;
        const Eigen::Vector3d tr = transform.translation();
        const Eigen::Quaterniond q(transform.rotation());
        tmsg.transform.translation.x = tr.x();
        tmsg.transform.translation.y = tr.y();
        tmsg.transform.translation.z = tr.z();
        tmsg.transform.rotation.x = q.x();
        tmsg.transform.rotation.y = q.y();
        tmsg.transform.rotation.z = q.z();
        tmsg.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(tmsg);
    }
    void publish_transform(const geometry_msgs::msg::TransformStamped& transform) const noexcept {
        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Node* node_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
} // namespace rose_nav