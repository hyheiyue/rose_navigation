cmake_minimum_required(VERSION 3.8)
project(rose_navigation)

if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT ROSE_NAV_IPO_SUPPORTED OUTPUT ROSE_NAV_IPO_ERROR)

set(ROSE_NAV_RELEASE_COMPILE_OPTIONS)
set(ROSE_NAV_RELEASE_LINK_OPTIONS)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND ROSE_NAV_RELEASE_COMPILE_OPTIONS
        -O3
        -DNDEBUG
        -march=native
        -ffast-math
        -fno-math-errno
        -fno-semantic-interposition
    )
    list(APPEND ROSE_NAV_RELEASE_LINK_OPTIONS
        -Wl,-O1
    )
endif()
# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)
find_package(livox_ros_driver2 QUIET)
if(livox_ros_driver2_FOUND)
    message(STATUS "Found livox_ros_driver2, enabling Livox lidar support.")
    add_definitions(-DHAVE_LIVOX_DRIVER)
else()
    message(WARNING "livox_ros_driver2 not found. Building without Livox lidar support.")
endif()
# find_package(robin REQUIRED)
# find_package(kiss_matcher REQUIRED)
set(OSQP_LIB_PATH /usr/local/lib/libosqp.so)  
include_directories(/usr/local/include/osqp)  
message(STATUS "Using manually specified OSQP C library: ${OSQP_LIB_PATH}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/param_deliver.h.in  
    ${CMAKE_BINARY_DIR}/src/param_deliver.h            
    @ONLY
)
file(GLOB_RECURSE SOURCE src/*.cpp 3rdparty/*.c 3rdparty/*.cpp)
ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_compile_options(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Release>:${ROSE_NAV_RELEASE_COMPILE_OPTIONS}>"
)
target_link_options(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Release>:${ROSE_NAV_RELEASE_LINK_OPTIONS}>"
)
if(ROSE_NAV_IPO_SUPPORTED)
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO is not supported: ${ROSE_NAV_IPO_ERROR}")
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
    3rdparty
    src
    ${CMAKE_BINARY_DIR}/src
)
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    yaml-cpp
    TBB::tbb
    OsqpEigen::OsqpEigen
    # robin::robin
    # kiss_matcher::kiss_matcher_core
    ${OpenCV_LIBS}
    ${OSQP_LIB_PATH}
    
)
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    rclcpp_components
    rclpy
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    tf2_geometry_msgs
    std_srvs
    visualization_msgs
    nav2_msgs
)
if(livox_ros_driver2_FOUND)
    ament_target_dependencies(${PROJECT_NAME}
        livox_ros_driver2
    )
endif()

rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN "rose_nav::SmallPointLioNode"
    EXECUTABLE ${PROJECT_NAME}_lm_node
)
rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN "rose_nav::RoseMapNode"
    EXECUTABLE ${PROJECT_NAME}_map_node
)

rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN "rose_nav::RosePlannerNode"
    EXECUTABLE ${PROJECT_NAME}_planner_node
)


ament_auto_package(
    INSTALL_TO_SHARE
    config
    launch
)
