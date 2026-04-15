from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    rose_nav_lm_node = Node(
        package="rose_navigation",
        executable="rose_navigation_lm_node",
        name="rose_navigation_lm_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("rose_navigation"),
                    "config",
                    "mid360_lm.yaml",
                ]
            )
        ],
        prefix='gnome-terminal -- bash -c "gdb -ex run --args $(ros2 pkg prefix rose_navigation)/lib/rose_navigation/rose_navigation_lm_node \
        --ros-args -r __node:=rose_navigation_lm_node \
        --params-file $(ros2 pkg prefix rose_navigation)/share/rose_navigation/config/mid360_lm.yaml; exec bash"'
    )
    static_base_link_to_livox_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--roll",
            "0.0",
            "--pitch",
            "0.0",
            "--yaw",
            "0.0",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "livox_frame",
        ],
    )

    return LaunchDescription([rose_nav_lm_node,static_base_link_to_livox_frame])