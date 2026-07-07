# rose_navigation 项目 SLAM 技术点面试总结

本文基于本项目源码整理，核心 SLAM/定位模块位于 `src/lm`，主要实现为 Small Point-LIO 风格的激光-惯性里程计与建图；`src/map` 将定位输出的点云进一步构建为占据地图和 ESDF，服务导航规划。

## 1. 项目 SLAM 总体架构

本项目的 SLAM 不是传统图优化后端框架，而是实时 LIO 前端为主：

1. 传感器输入：
   - LiDAR：支持 Livox、自定义 Mid360、Unitree、Velodyne 等适配器。
   - IMU：订阅角速度和线加速度。
2. 预处理：
   - 距离滤波：过滤过近、过远点。
   - 点数抽样：`point_filter_num`。
   - 体素降采样：`space_downsample_leaf_size`。
   - 按时间排序，并按 `batch_interval` 形成点云小批量。
3. 状态估计：
   - 使用 ESKF/Error-State Kalman Filter。
   - IMU 用于状态预测和角速度/加速度测量更新。
   - LiDAR 点云通过点到局部平面残差约束位姿。
4. 局部地图：
   - 使用 `SmallIVox` 稀疏体素哈希地图保存已配准点。
   - 每个体素保留一个代表点，并从当前体素及 6 邻域查找近邻。
5. 可选全局地图对齐：
   - 使用先验 PCD 和 small_gicp 做 GICP 对齐，发布 `map -> odom`。
6. 输出：
   - `/Odometry`：里程计。
   - `/cloud_registered`：配准点云。
   - TF：`odom -> base`，可选 `map -> odom`。
   - 可保存 PCD 地图。

面试一句话概括：

> 本项目实现的是轻量级激光惯性里程计。IMU 提供高频运动预测，LiDAR 点云在局部体素地图中做点到面匹配，通过 ESKF 融合约束并实时维护局部地图；如果加载先验 PCD，还能通过 GICP 估计 `map-odom` 修正，实现全局地图对齐。

## 2. ESKF 状态定义

代码位置：`src/lm/eskf.h`

状态维度为 30：

```text
x = [
  p, R, R_LI, t_LI,
  v, omega, a,
  g, b_g, b_a
]
```

含义：

- `p`：IMU/机器人在 odom 坐标系下的位置。
- `R`：IMU/机器人姿态。
- `R_LI, t_LI`：LiDAR 到 IMU 的外参，配置中 `extrinsic_est_en=false` 时固定。
- `v`：速度。
- `omega`：角速度状态。
- `a`：加速度状态。
- `g`：重力。
- `b_g`：陀螺仪零偏。
- `b_a`：加速度计零偏。

误差状态更新使用李群形式：

```text
p <- p + delta_p
R <- R Exp(delta_theta)
R_LI <- R_LI Exp(delta_theta_LI)
t_LI <- t_LI + delta_t_LI
v <- v + delta_v
omega <- omega + delta_omega
a <- a + delta_a
g <- g + delta_g
b_g <- b_g + delta_b_g
b_a <- b_a + delta_b_a
```

其中 SO(3) 指数映射为 Rodrigues 公式：

```text
Exp(phi) = I + sin(theta) K + (1 - cos(theta)) K^2
theta = ||phi||
K = hat(phi / theta)
```

`hat(v)` 是反对称矩阵：

```text
hat([x,y,z]^T) =
[  0 -z  y
   z  0 -x
  -y  x  0 ]
```

面试重点：

- 姿态不能简单向量相加，代码用 `R * Exp(delta_theta)` 在 SO(3) 上更新。
- 误差状态滤波维护的是小扰动，适合非线性位姿估计。

## 3. IMU 预测模型

代码位置：`eskf::predict_state()`、`eskf::predict_cov()`

状态预测：

```text
p_{k+1} = p_k + v_k dt
R_{k+1} = R_k Exp(omega_k dt)
v_{k+1} = v_k + (R_k a_k + g_k) dt
```

协方差预测：

```text
P_{k+1} = F P_k F^T + Q dt^2
```

其中关键雅可比包含：

```text
F_pv = I dt
F_RR = Exp(-omega dt)
F_Romega = A(-omega dt) dt
F_vR = -R hat(a)
F_va = R dt
F_vg = I dt
```

`Q` 来自配置中的：

- `velocity_cov`
- `acceleration_cov`
- `omg_cov`
- `ba_cov`
- `bg_cov`

IMU 测量更新：

代码位置：`Estimator::h_imu()`、`eskf::update_imu()`

残差：

```text
r_gyro = omega_meas - omega - b_g
r_acc  = s_acc * acc_meas - a - b_a
```

其中：

```text
s_acc = ||gravity|| / acc_norm
```

作用：

- 用 IMU 测量约束内部角速度、加速度和零偏。
- 配置支持饱和检测：超过 `satu_gyro` 或 `satu_acc` 时跳过对应维度。

初始化重力方向：

若 `fix_gravity_direction=true`，系统等待约 200 个 IMU 数据，取加速度均值估计重力方向，并把模长缩放为配置重力大小：

```text
g = - ||g_config|| * mean(acc) / ||mean(acc)||
a = -g
```

面试回答：

> IMU 在系统中有两个作用：一是高频传播位姿、速度和协方差，保证点云扫描期间能按时间预测；二是作为测量更新角速度、加速度及零偏，降低纯 LiDAR 匹配的抖动和退化风险。

## 4. LiDAR 点到局部平面匹配

代码位置：`Estimator::h_point()`、`Estimator::h_batch()`

当前配置 `batch_update: true`，主要走批量更新。

### 4.1 点坐标变换

LiDAR 点先变换到 IMU/机器人坐标系：

```text
p_I = R_LI p_L + t_LI
```

批量更新中考虑点在小批次内的时间偏移 `dt`：

```text
R_delta = Exp(omega dt)
p_W = (R R_delta) p_I + p + v dt
```

逐点更新中可简化为：

```text
p_W = R p_I + p
```

这里的 `W` 对应 odom/map 内部坐标。

### 4.2 近邻搜索与平面拟合

局部地图用 `SmallIVox` 存储点。当前点变换到 odom 后，在体素哈希地图中取近邻点，代码默认：

```text
NUM_MATCH_POINTS = 100
MIN_MATCH_POINTS = 5
```

平面拟合方法是 PCA：

```text
c = (1/N) sum_i q_i
Sigma = (1/(N-1)) sum_i (q_i - c)(q_i - c)^T
```

对协方差矩阵求特征分解，最小特征值对应的特征向量为平面法向量：

```text
n = eigenvector_min(Sigma)
d = -n^T c
```

平面方程：

```text
n^T x + d = 0
```

点到平面有符号距离：

```text
r = n^T p_W + d
```

代码中的测量残差设为：

```text
z = -r
```

并用 `plane_threshold` 检查近邻点是否确实近似共面。

### 4.3 点到面残差雅可比

对位姿小扰动，残差：

```text
r = n^T (R p_I + p) + d
```

对位置的雅可比：

```text
dr/dp = n^T
```

对姿态扰动的雅可比，代码写法为：

```text
A = p_I x (R^T n)
```

因此测量矩阵主要为：

```text
H = [ n^T, A^T, 0, 0 ]
```

如果开启外参估计 `extrinsic_est_en=true`，还会加入：

```text
C = R^T n
A = p_I x C
B = p_L x (R_LI^T C)
H = [ n^T, A^T, B^T, C^T ]
```

面试重点：

- LiDAR 不是直接点到点 ICP，而是点到局部平面约束。
- 平面来自局部地图近邻 PCA。
- 点到面残差对结构化场景更稳定，计算量也低。

## 5. 批量 LiDAR 更新的信息形式

代码位置：`eskf::update_iterated_batch()`

批量更新把多个点残差累积成正规方程。每个点：

```text
z_i = H_i delta_x + noise
R_i = laser_point_cov
w_i = count_i / R_i
```

累积：

```text
A = sum_i H_i^T w_i H_i
b = sum_i H_i^T w_i z_i
```

先验信息矩阵：

```text
Lambda = P^{-1}
```

求解：

```text
(Lambda + A) delta_x = b
delta_x = (Lambda + A)^{-1} b
```

更新状态：

```text
x <- x plus delta_x
P <- (Lambda + A)^{-1}
```

代码中只对前 12 维构建点云观测雅可比，即：

```text
[p, R, R_LI, t_LI]
```

面试说法：

> 批量更新本质上是把一批点到面残差转成加权最小二乘，并和 ESKF 先验信息矩阵融合。这样比逐点 Kalman 更新更适合点云批处理，也方便并行计算残差。

## 6. 局部体素地图 SmallIVox

代码位置：`src/lm/small_ivox.h`

实现要点：

- 坐标转体素索引：

```text
key = floor(p / resolution)
```

- 哈希编码：

```text
hash = key_x << 32 | key_y << 16 | key_z
```

- 每个体素只保存一个代表点。
- 使用链表维护缓存，超过容量后移除最旧体素。
- 近邻查询搜索当前体素和 x/y/z 六邻域。

优点：

- 查询复杂度接近 O(1)。
- 自动降采样，控制地图规模。
- 适合实时 LIO。

局限：

- 每个体素只保留一个点，细节会被压缩。
- 只查 6 邻域，强依赖体素分辨率和点云密度。
- 没有回环检测，长时间运行仍可能累计漂移。

## 7. 点云预处理与运动补偿

代码位置：`SmallPointLIO::Impl::on_point_cloud_callback()`、`handle_once()`

预处理包括：

```text
min_distance <= ||p|| <= max_distance
i % point_filter_num == 0
voxelgrid_sampling_tbb(...)
按 timestamp 排序
按 batch_interval 分批
```

批量点云更新中，代码对批内每个点使用：

```text
dt = point.timestamp - batch.timestamp
R_delta = Exp(omega dt)
p_W = (R R_delta) p_I + p + v dt
```

这相当于对扫描内运动做一阶补偿，减轻运动畸变。

面试回答：

> Livox/Mid360 这类雷达点有时间戳，如果直接整帧当同一时刻，会在快速运动时产生畸变。本项目用当前角速度和速度对每个点按时间偏移进行位姿外推，把点投到统一坐标系后再做匹配。

## 8. 先验 PCD 地图与 GICP 对齐

代码位置：`SmallPointLIO::Impl::algin_callback()`

配置项：

- `use_priori_pcd_add_ivox`：将先验 PCD 加入 LIO 局部体素地图。
- `use_priori_pcd_for_algin`：使用先验 PCD 做 GICP 对齐。
- `prior_pcd_path`：先验点云路径。
- `init_pose_in_prior_pcd`：初始位姿。

GICP 基本思想：

传统 ICP 最小化点到点距离：

```text
min_T sum_i || q_i - T p_i ||^2
```

GICP 为每个点估计局部协方差，使用马氏距离：

```text
min_T sum_i d_i^T (C_i^target + R C_i^source R^T)^{-1} d_i
d_i = q_i - T p_i
```

代码流程：

1. 读取先验 PCD 为 target cloud。
2. 对 target 做体素采样、法向估计、协方差估计、KDTree。
3. 运行时积累当前注册点云为 source cloud。
4. 用 small_gicp 对齐 target/source。
5. 得到 `T_target_source`，更新 `now_pose_in_prior_pcd_`。
6. 高频发布 `map -> odom`。

面试回答：

> LIO 负责相对运动估计，但会漂移。加载先验 PCD 后，系统用 GICP 将当前累计点云对齐到先验地图，得到 `map-odom` 变换，相当于把局部里程计约束到全局地图坐标。

注意：源码里 `reset_trigger_` 和 `aligin_trigger_` 都注册了 `"reset"` 服务名，这可能导致服务名冲突，面试时如果被问到工程风险，可以主动指出。

## 9. 占据地图与 ESDF

代码位置：`src/map/occ_map.cpp`、`src/map/esdf.cpp`

这部分不是 LIO 的状态估计核心，但用于把 SLAM 点云输出转成导航可用地图。

### 9.1 Log-Odds 占据栅格

占据概率 `p` 转 log-odds：

```text
L = log(p / (1 - p))
```

命中更新：

```text
L <- clamp(L + log_hit, log_min, log_max)
```

空闲更新：

```text
L <- clamp(L + log_free, log_min, log_max)
```

判断占据：

```text
occupied = L > occ_th
```

项目中还结合 `last_update` 和 `timeout`，过期栅格不再视为有效占据。

### 9.2 Raycasting 空闲更新

传感器原点到命中点之间的体素应更新为空闲。代码使用 DDA raycast：

```text
origin voxel -> hit voxel
沿射线遍历经过体素，作为 free cells
hit voxel 作为 occupied cell
```

DDA 的作用是高效计算射线穿过哪些体素，避免连续空间采样。

### 9.3 ESDF

ESDF 表示每个栅格到最近障碍物的符号距离：

```text
ESDF(x) = dist_to_occ(x) - dist_to_free(x)
```

代码用两次距离传播：

- `dist_to_occ`：到最近占据格的距离。
- `dist_to_free`：到最近空闲格的距离。

最后：

```text
esdf = dist_to_occ - dist_to_free
```

含义：

- 自由空间通常为正距离。
- 障碍物内部或占据区域为负距离。
- 规划器可用 ESDF 查询障碍距离和安全裕度。

## 10. 为什么这个算法能实时

可从四个角度回答：

1. 数据降维：
   - 距离过滤、点数抽样、体素降采样。
2. 局部地图：
   - `SmallIVox` 哈希体素近邻查询接近 O(1)。
3. 批量并行：
   - 点到面残差构建使用 TBB 并行。
   - 体素采样、GICP 法向/协方差估计也使用 TBB。
4. 滤波而非全局优化：
   - ESKF 只维护当前状态和协方差，不做大规模位姿图优化。

## 11. 可能的面试追问与回答

### Q1：这个项目用的是哪类 SLAM 算法？

答：

> 是 LiDAR-Inertial Odometry and Mapping，类似 Point-LIO 的滤波式 LIO。IMU 负责高频预测，LiDAR 点云通过点到局部平面残差更新 ESKF，同时维护体素局部地图。它更偏实时里程计建图，不是带回环的全局图优化 SLAM。

### Q2：为什么要用 ESKF？

答：

> 位姿在 SE(3)/SO(3) 上是非线性的，尤其姿态不能直接欧式相加。ESKF 把真实状态表示为名义状态加小误差，姿态误差通过 `Exp(delta_theta)` 注入，既保持了滤波效率，也能正确处理旋转流形。

### Q3：LiDAR 残差怎么构造？

答：

> 当前点先通过外参和当前位姿投到 odom 坐标系，然后在局部体素地图中找近邻，对近邻做 PCA 拟合平面。最小特征值对应的特征向量作为平面法向，点到平面的有符号距离 `n^T p + d` 就是残差，再对位置和姿态求雅可比进入滤波更新。

### Q4：IMU 在滤波里怎么用？

答：

> 一方面按 `p=p+vdt`、`R=R Exp(omega dt)`、`v=v+(Ra+g)dt` 做状态预测和协方差传播；另一方面用 IMU 角速度、加速度作为观测，残差分别是 `omega_meas - omega - b_g` 和 `acc_meas - a - b_a`，用于估计角速度、加速度和零偏。

### Q5：批量点云更新和逐点更新有什么区别？

答：

> 逐点更新是每个点单独做一次 Kalman update；批量更新把一批点的 `H^T R^-1 H` 和 `H^T R^-1 z` 累加成一个线性系统，再和先验信息矩阵合并求解。当前配置使用批量更新，优势是可以并行计算残差，整体更适合高频点云。

### Q6：如何处理雷达运动畸变？

答：

> 每个点保留时间戳，批量更新中用点时间与批次平均时间的差 `dt`，通过 `Exp(omega dt)` 和 `v dt` 对点位姿做外推，近似把扫描内不同时间的点补偿到统一时刻。

### Q7：局部地图怎么维护？

答：

> 用哈希体素地图 `SmallIVox`。世界坐标除以分辨率后 floor 得到体素 key，再哈希成整数索引。每个体素只存一个代表点，超过容量时淘汰旧点。匹配时查当前体素和 6 邻域，快速获得近邻点做平面拟合。

### Q8：有没有全局定位或回环？

答：

> 没有传统回环检测和位姿图优化。但项目支持先验 PCD 地图对齐：用 small_gicp 将当前累计点云和先验地图匹配，估计 `map-odom` 变换，用于把局部 LIO 约束到全局地图。

### Q9：占据地图和 ESDF 是怎么来的？

答：

> LIO 输出注册点云后，地图模块把点云插入滑动体素地图。命中点增加 log-odds，射线经过的体素降低 log-odds 表示空闲。之后根据二值占据信息做距离传播，生成 ESDF，规划器可以查询到障碍物的距离。

### Q10：这个方案的优缺点？

优点：

- 实时性好，适合机器人在线定位建图。
- IMU 和 LiDAR 互补，短时运动预测稳定。
- 点到面约束比点到点更适合结构化环境。
- 体素哈希和 TBB 并行降低计算开销。
- 支持先验 PCD 地图对齐。

不足：

- 无回环检测，长期运行仍有漂移。
- 对几何退化场景敏感，如长走廊、平面少、动态物体多。
- 外参默认固定，标定误差会影响精度。
- 体素地图每格一个代表点，细节表达有限。
- GICP 对齐依赖初值和先验地图质量。

## 12. 可直接背诵的 1 分钟版本

> 本项目的 SLAM 核心是一个轻量级 LiDAR-Inertial Odometry。系统订阅 LiDAR 点云和 IMU，先对点云做距离滤波、抽样、体素降采样并按时间分批。状态估计采用 ESKF，状态包含位置、姿态、速度、角速度、加速度、重力、IMU 零偏以及可选 LiDAR-IMU 外参。IMU 用运动模型进行高频预测，公式是 `p=p+vdt`、`R=R Exp(omega dt)`、`v=v+(Ra+g)dt`，同时用角速度和加速度残差估计零偏。LiDAR 更新时，将点通过外参和当前位姿投到 odom 坐标系，在 SmallIVox 体素哈希局部地图中找近邻，用 PCA 拟合局部平面，构造点到面残差 `n^T p + d`，再把一批点的残差累积成加权最小二乘，与 ESKF 先验信息矩阵融合更新状态。地图方面，系统实时把配准点加入局部体素地图，并可保存 PCD；如果加载先验 PCD，还能用 GICP 对齐当前点云和先验地图，发布 `map-odom` 变换。导航层再用点云构建 log-odds 占据地图和 ESDF，供规划器查询障碍距离。

## 13. 源码定位速查

- `src/lm/eskf.h`：ESKF 状态、预测、IMU 更新、点云批量更新。
- `src/lm/estimator.cpp`：LiDAR 点到面残差、近邻平面拟合、IMU 残差。
- `src/lm/small_point_lio.cpp`：ROS2 订阅发布、数据预处理、时间同步、主流程、GICP 对齐。
- `src/lm/small_ivox.h`：稀疏体素哈希局部地图。
- `src/lm/so3_math.h`：SO(3) 指数映射、hat 矩阵、A 矩阵。
- `config/mid360_lm.yaml`：Mid360 LIO 参数。
- `src/map/occ_map.cpp`：log-odds 占据地图、raycasting。
- `src/map/esdf.cpp`：ESDF 距离场构建。

