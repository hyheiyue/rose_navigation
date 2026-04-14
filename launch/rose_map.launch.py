from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    rose_nav_map_node = Node(
        package="rose_navigation",
        executable="rose_navigation_map_node",
        name="rose_navigation_map_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("rose_navigation"),
                    "config",
                    "rose_map.yaml",
                ]
            )
        ],
        # prefix='gnome-terminal -- bash -c "gdb -ex run --args $(ros2 pkg prefix rose_navigation)/lib/rose_navigation/rose_navigation_map_node \
        # --ros-args -r __node:=rose_navigation_map_node \
        # --params-file $(ros2 pkg prefix rose_navigation)/share/rose_navigation/config/rose_map.yaml; exec bash"'
    )
  

    return LaunchDescription([rose_nav_map_node])