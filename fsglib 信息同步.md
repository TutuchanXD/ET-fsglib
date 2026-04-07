# `fsglib` 信息同步

## 1. 文档用途

本文档作为 `fsglib` 项目的长期记忆文档，用于：

- 记录用户侧已确认的输入信息与约束
- 记录当前实现阶段的关键工程决策
- 记录仿真端与解算端的一致性核查结果
- 记录后续需要继续确认或补充的信息

建议维护方式：

- 用户有新信息时，直接在对应章节补充或修订
- Codex 在完成关键核查、定位出重要问题、或形成新的工程约定后，补充到“最新核查结论”或“待办事项”中
- 若某条信息已失效，不删除原文，可在其后补充“更新说明”或“当前结论”

---

## 2. 用户已同步的原始信息

### 2.1 探测器安装关系

Q：每片导星探测器相对于望远镜本体系的安装关系：探测器法线、行方向、列方向、中心点位置，最好直接给旋转矩阵。
A：这个仍不清楚，是必须的吗？能否按照最理想假设来。

当前结论：

- 第一阶段可按理想安装假设推进
- 接口必须预留安装矩阵 / 外参输入，后续拿到真实安装关系后可直接替换

### 2.2 探测器内参

Q：每片导星探测器的内参：`dx, dy, u0, v0, f` 或 `fx, fy, u0, v0`。A：

- `dx = dy = 6.5e-3 mm/pixel`
- `f ≈ 426 mm`

理想情况下，以光轴为 `(0, 0)` 建立坐标系，单位为 `mm`。

4 片主探测器视场边缘范围：

```python
{
    "main_ul": {"x": (-92.95, -1.75), "y": (1.97, 90.97)},
    "main_ur": {"x": (1.97, 90.97), "y": (1.75, 92.95)},
    "main_lr": {"x": (1.75, 92.95), "y": (-90.97, -1.97)},
    "main_ll": {"x": (-90.97, -1.97), "y": (-92.95, -1.75)},
}
```

4 片导星探测器视场边缘范围：

```python
{
    "guide_ul": {"x": (-40.66, -27.34), "y": (102.91, 116.22)},
    "guide_ur": {"x": (102.91, 116.22), "y": (27.34, 40.66)},
    "guide_lr": {"x": (27.34, 40.66), "y": (-116.22, -102.91)},
    "guide_ll": {"x": (-116.22, -102.91), "y": (-40.66, -27.34)},
}
```

补充说明：

- 主探测器像元大小：`10 um`
- 导星探测器像元大小：`6.5 um`
- 主探测器已知比例尺：`4.83 arcsec/pix`（对应 `10 um/pix`）

### 2.3 场点到像面的映射

Q：每片导星探测器覆盖区域内的场点到像面映射，提供 Zemax 导出的网格 LUT，或提供畸变模型系数。
A：基于同事提供的主探测器边缘点拟合得到：

```text
u ≈ 0.134636212 x - 3.1445e-7 x^3 - 8.5532e-8 x y^2
v ≈ 0.134636212 y - 3.1445e-7 y^3 - 8.5532e-8 y x^2
```

其中：

- `x, y` 单位为 `mm`
- `u, v` 单位为 `deg`

相关 notebook：

- `/home/cxgao/ET/FOV/xy2uv/xy2uv.ipynb`

### 2.4 局部视轴 / 主光线方向

Q：每片导星探测器对应的局部视轴 / 主光线方向。
A：这一条没看懂，前面是不是已经解决了？

当前结论：

- 目前尚未得到严格定义
- 工程上可先等价为探测器安装矩阵与局部投影中心共同决定

### 2.5 PSF 数据

Q：每片导星探测器在 `465–940 nm` 宽谱、工作温度下的 PSF 数据，最好按场点网格给出。A：

- PSF 数据：`/home/cxgao/ET/Photosim6/data/psf/et/241006/D280mm-focus/sim_psf_images.pkl`
- 说明文档：`/home/cxgao/ET/Photosim6/data/psf/et/241006/D280mm-focus/sim_psf_images_structure_note.md`

### 2.6 温度 / 离焦变化

Q：温度变化或离焦变化下的映射 / PSF 变化。
A：目前没有温度变化数据，但是要保留接口，后面有数据我们直接用。

### 2.7 Gaia 星表

Q：Gaia 测试星表。A：

- 根目录：`/home/cxgao/gaia_dr3_19mag`
- 为 `Gaia DR3` 全部 `19` 等以内星表
- 以 healpix 天区划分
- `Npix = 12 * NSIDE^2 = 12 * 32^2 = 12288`

补充提醒：

- 当前目录总量很大，约 `88 GB`
- 后续运行时不应默认“全目录暴力扫描”

### 2.8 参数标准化

Q：参数输入要标准化，即使现在一些输入还没有确定，我们仍要留出接口。
A：同意。

### 2.9 布局图

主探测器和导星探测器的分布图已提供，供参考。

### 2.10 当前仿真数据的说明

用户说明：

- 当前仿真星图位于 `/home/cxgao/ET/FSG_images_sims/v2`
- 这些图是基于实际星场生成的
- 文件名中写入的坐标中心目前应理解为主探测器坐标中心
- 当前这批数据主要用于算法链路验证
- 后续拿到导星探测器真实测试参数后，应能快速替换验证

相关仿真参数文件：

- `/home/cxgao/ET/Photosim6ft/et_100_det_inputs_1h.xlsx`

### 2.11 星点提取策略

用户说明：

- 当前阶段星点提取可先采用一阶质心
- 但必须保留后续切换到更高精度方法的接口
- 若一阶质心无法达到目标精度，应继续优化

### 2.12 Lost-in-Space 先验

用户说明：

- 文档流程中虽提到基于星敏感器初始姿态进入 Lost-in-Space / initial acquisition
- 但当前手头没有星敏团队提供的任何数据

### 2.13 当前运行环境

用户说明：

- 当前先在 conda `base` 环境中运行
- 等完整开发落地后，再配置专有环境

---

## 3. 当前开发阶段约定

### 3.1 当前主验证对象

当前第一优先验证对象是 **凌星相机精度闭环**。

微引力导星相机指标也已记录，但暂不作为第一阶段主验收项。

### 3.2 当前 `v2` 数据的角色

当前 `v2` 数据在 `fsglib` 中按如下口径使用：

- 视作“主探测器单探测器验证基线”
- 图像尺寸：`1119 x 1119`
- 比例尺：`4.83 arcsec/pix`
- 每个 batch：`10000` 帧
- 单帧曝光 / cadence：`10 s`

已知 4 个 FIELD CENTER：

```python
FIELD_CENTERS_DEG = [
    (304.098, 51.433),
    (294.179, 41.107),
    (292.559, 54.737),
    (287.276, 47.686),
]
```

### 3.3 当前实现策略

当前 `fsglib` 阶段性策略：

- 单帧 `init` 与多帧 `tracking` 已能闭环
- 当前主链路默认使用简单质心提取
- 更高精度的 PSF-aware 质心 / 模板拟合仍需后续补上
- 主探测器仿真几何当前使用 `sky_patch_linearized` 口径

---

## 4. 精度目标

### 4.1 凌星导星相机

- 单帧精度：`<= 0.02"`（1σ，非绕光轴），`<= 0.05"`（1σ，绕光轴）
- `1 min` 多帧叠加精度：`<= 0.003"`（1σ，非绕光轴），`<= 0.01"`（1σ，绕光轴）

### 4.2 微引力导星相机

- 单帧精度：`<= 0.03"`（1σ，非绕光轴），`<= 0.8"`（1σ，绕光轴）
- `10 min` 多帧叠加精度：`<= 0.003"`（1σ，非绕光轴），`<= 0.1"`（1σ，绕光轴）

---

## 5. 最新核查结论

### 5.1 参考星来源一致性

已确认：

- `v2` 仿真端使用的星源与 `fsglib` 当前使用的星源在根本上是一致的，都是 `Gaia DR3`
- `stars.ecsv` 中的星是从仿真端 Gaia 查询结果中裁剪得到的子集
- 对 `v2` 的 4 个 batch 逐一核对后，`stars.ecsv` 中的 `200` 颗星都能在 `fsglib` 当前 Gaia healpix 后端查询结果中找到

当前结论：

- **不存在“仿真端按星表 A 仿真、解算端按完全不同的星表 B 查询”的问题**
- 但当前仍存在“同源星表下，选星口径与筛选口径不一致”的问题

### 5.2 仿真端与 `fsglib` 当前选星口径不一致

仿真端当前做法：

- 使用矩形 sky patch 查询 [field.py](/home/cxgao/ET/Photosim6ft/photsim6ft/field.py#L406)
- 查询星等上限为 `Gaia G <= 15`
- 再按 `Kepler Mag` 从亮到暗裁成最亮 `200` 颗 [et_sim_100_det.py](/home/cxgao/ET/Photosim6ft/et_sim_100_det.py#L1255)

`fsglib` 当前做法：

- 使用圆锥查询 [catalog.py](/home/cxgao/ET/FSG/fsglib/fsglib/ephemeris/catalog.py#L21)
- 查询半径当前仍按角距离圆锥处理
- 使用 `Gaia G` 星等，不使用仿真端的 `Kepler Mag` 口径
- 不裁到仿真端相同的 `200` 颗参考星

直接影响：

- 仿真图中真实参与成像的星约 `200` 颗
- `fsglib` 当前构造的参考星可达数千到上万颗
- 这会显著放大匹配空间、耗时和误匹配风险

### 5.3 `v2` 当前并非“纯 10.2–11.2 Mv 导航星”

对当前 `stars.ecsv` 检查后发现：

- `v2` 当前数据并不只是严格的 `10.2–11.2` 星等导航星
- 例如：
  - `batch0` 的 `Kepler Mag` 范围约为 `5.89 ~ 14.44`
  - `batch1` 的 `Kepler Mag` 范围约为 `7.29 ~ 13.28`

这与“先查 Gaia G，再转 Kepler Mag，并保留最亮 200 颗”的当前仿真逻辑一致。

### 5.4 当前 guide 首帧的 detector 坐标口径已拆清

对 `/home/cxgao/ET/FSG_guide_sims/guide_det_v1_noise_psf_6s` 的首帧逐批核查后，当前已确认：

- `run_meta.json` 中脚本层 `apply_static_field_offset = False`
- `run_meta.json` 中 `field_offset_x_pix = field_offset_y_pix = 0.0`
- 但 `npz["truth_x_detector_pix"] / ["truth_y_detector_pix"]` 与 `stars.ecsv["Detector Xpix/Ypix"]` 之间仍存在稳定常量偏移

当前这批 guide 数据中，该常量偏移为：

- `dx ≈ +0.1326333675 pix`
- `dy ≈ -0.0500387349 pix`
- 径向 RMS `≈ 0.1417585453 pix`

并且该偏移在 4 个 batch 上都一致，批内散布只有 `1e-12 pix` 量级，可视为严格常量项。

### 5.5 `telescope FOV offset` 的真实来源

当前已确认，这个常量偏移并不是脚本层 `field_offset_x/y`，而是 `photsim6ft` 仪器层的 telescope 级随机 FOV offset：

- [instrumentation.py](/home/cxgao/ET/Photosim6ft/photsim6ft/instrumentation.py#L108) 中，`Telescope` 初始化时先设 `fov_xy_offset_pix = (0, 0)`
- 当 `fov_offset_max > 0` 时，会在 [instrumentation.py](/home/cxgao/ET/Photosim6ft/photsim6ft/instrumentation.py#L114) 中随机采样一个整批固定的 `x/y` 偏移
- [configurator.py](/home/cxgao/ET/Photosim6ft/photsim6ft/configurator.py#L103) 中该参数默认值是 `Telescope FOV Max Offset = 1.0 pix`
- 当前 [et_sim_guide_det_v1_noise_psf.py](/home/cxgao/ET/Photosim6ft/et_sim_guide_det_v1_noise_psf.py) 没有显式把这个参数置零，因此默认值仍会生效

该 telescope offset 后续被写进了 NPZ detector truth：

- [frame_truth.py](/home/cxgao/ET/Photosim6ft/photsim6ft/frame_truth.py#L75) 读取 `self.telescope.fov_xy_offset_pix`
- [frame_truth.py](/home/cxgao/ET/Photosim6ft/photsim6ft/frame_truth.py#L368) 到 [frame_truth.py](/home/cxgao/ET/Photosim6ft/photsim6ft/frame_truth.py#L374) 将该 offset 加入 `truth_static_x/y_detector_pix` 与 `truth_x/y_detector_pix`

当前关键点：

- 该项不是用户在仿真脚本里显式添加的 static field offset
- 该项当前也没有写进 `run_meta.json`
- 因而如果只看 `run_meta`，会误以为“完全没有 offset”

### 5.6 `fsglib` 当前 guide 几何主链已经改成 exact `et_focalplane`

当前 guide 首帧主链已经不再使用 `body_model_proxy` 生成观测 `los_body`。

当前实现策略是：

- 观测侧 `detector pixel -> pixel_to_sky() -> body rotation`
- 参考侧 `query_detector_sources()` 直接给出 raw detector 预测像点

相关实现：

- [transform.py](/home/cxgao/ET/et_focalplane/src/et_coord/transform.py#L86)
- [run_guide_init.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_init.py#L269)
- [run_guide_truth_noise.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_truth_noise.py#L123)

核查结果：

- `sim_to_detector_map_error_pix` RMS 约 `1e-13 pix`
- `match_predicted_vs_ecsv_detector_pix` RMS 约 `2.7e-12 pix`

这说明：

- `image -> detector` 桥接当前已基本无误
- `et_focalplane` 的 raw detector 预测坐标与 `stars.ecsv` 是严格一致的

### 5.7 当前推荐的姿态精度口径

当前 guide 首帧报告应同时保留两套姿态差，但主指标已经明确：

1. `current_to_frame_truth`

- 含义：当前解相对于“实际仿真帧真值”的姿态误差
- 这是当前最推荐的主指标
- 用它汇报“算法在仿真帧上到底做得怎样”

2. `frame_truth_to_nominal_body`

- 含义：实际仿真帧相对于名义 `et_focalplane/body` 几何的固定偏移
- 当前主要反映 simulator 的 telescope FOV offset
- 不应再把它和算法噪声混在一起解释

说明：

- 旧的 `current_to_oracle` 在当前 guide exact 链路下不再适合作为首选主指标，因为它把上述两项混合在了一起
- notebook 与示例脚本当前都已切到这套新口径

相关 notebook：

- [guide_first_frame_review.ipynb](/home/cxgao/ET/FSG/fsglib/guide_first_frame_review.ipynb)
- [guide_first_frame_real_centroid_review.ipynb](/home/cxgao/ET/FSG/fsglib/guide_first_frame_real_centroid_review.ipynb)

### 5.8 理想质心首帧结果（`0.065 pix` 假设，exact `et_focalplane`）

当前实验口径：

- truth detector centroid
- 每轴注入 `N(0, 0.065^2) pix`
- exact `et_focalplane` 几何
- QUEST 姿态解算

首帧结果：

- `num_matched = 400`
- `solution residual RMS = 0.2627715313 arcsec`
- `centroid RMS = 0.0880314227 pix`
- `current_to_frame_truth = 0.0284769211 arcsec`
- 其中：
  - `non-roll = 0.0179219270 arcsec`
  - `roll = 0.0222120552 arcsec`
- `frame_truth_to_nominal_body = 0.4328407728 arcsec`

当前解释：

- 在 exact `et_focalplane` 口径下，`0.065 pix` 假设本身对应的“实际仿真帧姿态误差”已经很低
- 当前 `0.43"` 量级的大头不是算法噪声，而是仿真帧相对名义 geometry 的固定偏移

### 5.9 真实质心首帧结果（weighted centroid，exact `et_focalplane`）

当前实验口径：

- 实际仿真图像
- `weighted centroid`
- exact `et_focalplane` 几何
- QUEST 姿态解算

首帧结果：

- `num_matched = 390`
- `solution residual RMS = 1.8033697979 arcsec`
- `centroid RMS = 0.6513993755 pix`
- `current_to_frame_truth = 0.5153280974 arcsec`
- 其中：
  - `non-roll = 0.4856848057 arcsec`
  - `roll = 0.1722496390 arcsec`
- `frame_truth_to_nominal_body = 0.4317657309 arcsec`

当前解释：

- 当前 guide 首帧相对于“实际仿真帧真值”的主要误差源已经回到质心提取
- 几何主链本身已不再是主瓶颈

### 5.10 当前瓶颈与剩余问题

当前首帧链路下，最重要的判断是：

- 对“实际仿真帧真值”的姿态精度，当前主瓶颈是质心误差
- 对“名义 geometry”的固定偏移，当前主来源是 simulator telescope FOV offset 没有显式写入元数据
- `roll` 当前反而小于 `non-roll` 并不异常，因为 4 片导星环绕布局本身就更容易约束 roll，剩余固定偏移更像是 boresight / 非绕光轴方向的整体偏置

当前工程建议：

- 继续深入做质心误差链路分析
- 同时推动仿真端把 telescope FOV offset 显式写入 `run_meta.json`
- 后续所有报告都应明确区分“当前帧真值误差”和“名义 geometry 偏移”

---

## 6. 当前优先待办

### 6.1 必须先对齐的底层口径

1. 报告与 notebook 统一改用 `current_to_frame_truth` 作为首选姿态精度指标：

   - `frame_truth_to_nominal_body` 单独保留
   - 不再把旧的 `current_to_oracle` 当作主指标
2. 必须在仿真端显式导出 telescope FOV offset：

   - 最好把 `telescope.fov_xy_offset_pix["x/y"]` 写入 `run_meta.json`
   - 这样 `NPZ detector truth` 与 raw detector 口径的差异就能被显式解释
3. 评估层必须明确区分两套 detector 坐标：

   - `stars.ecsv["Detector Xpix/Ypix"]` 是 raw detector 坐标
   - `npz["truth_x/y_detector_pix"]` 当前包含了 telescope FOV offset
   - 若直接比较 `predicted_xy` 与 NPZ detector truth，必须注明口径

### 6.2 后续精度提升主线

1. 继续做真实质心误差场分析，明确不同 SNR / 位置 / PSF 条件下的误差来源
2. 若简单质心仍不够，再引入 PSF-aware centroid / 模板拟合
3. 在多帧链路和 Monte Carlo 里继续沿用 `current_to_frame_truth / frame_truth_to_nominal_body` 这套双口径输出

---

## 7. 需要用户后续补充或确认的信息

若后续能补充，优先级最高的是：

1. 导星探测器真实安装矩阵 / 外参
2. 导星探测器真实主点定义
3. 仿真端是否可以导出逐帧星位真值
4. 仿真端是否可以显式导出当前 batch 的静态 `x/y` 亚像元 offset
5. 当前 `v2` 数据是否就是后续都将沿用的主探测器验证基线，还是会继续更新版本

---

## 8. 更新记录

- `2026-03-19`
  - 重新整理为结构化记忆文档
  - 记录了 `v2` 仿真与 `fsglib` 当前一致性核查结果
  - 补充了凌星相机 / 微引力导星相机精度指标
  - 明确记录：星源同源，但选星口径、几何 offset、逐帧 truth 口径仍未完全对齐
  - 已在仿真脚本中把静态 field offset 做成显式开关，并计划写入 `run_meta.json`
  - `fsglib` 评估输出将区分 `non-roll / roll / total attitude` 三类姿态误差指标
- `2026-03-20`
  - 已基于最新重跑的 `v2` 仿真重新验证：4 个 batch 的 `run_meta.json` 均记录 `apply_static_field_offset=False`，且 `field_offset_x_pix = field_offset_y_pix = 0.0`
  - 用 `stars.ecsv` 的 `RA/Dec` 经当前 `fsglib` projector 回投到像面后，4 个 batch 的几何误差均约 `1e-11 pix`，说明“仿真静态几何 -> fsglib 静态几何”已基本完全对齐
  - 当前 4 个 batch 首帧 `init` 仍能稳定得到有效解，残差 RMS 约 `1.42" - 1.70"`
  - 但新仿真下的姿态误差已更清晰暴露：`non-roll` 约稳定在 `~3"`，`roll` 在不同 batch 间波动较大（约 `0.5" - 20"`），说明当前主瓶颈更像是图像端星点定位/亮度分布导致的旋向不稳定，而不再是静态几何口径问题
  - 4 个 batch、每 batch 2 帧的当前数据集级汇总：
    - `mean_rms_arcsec ≈ 1.578`
    - `mean_non_roll_error_arcsec ≈ 3.198`
    - `mean_roll_error_arcsec ≈ 8.635`
    - `mean_total_attitude_error_arcsec ≈ 9.879`
  - 仿真端 / `fsglib` 考虑点再次核查：
    - 当前 `main_sim_v2` 在 `fsglib` 中使用 `sky_patch_linearized` 投影，不启用畸变模型；这与当前仿真端“主探测器小视场线性成像”的几何口径是一致的
    - `fsglib` 代码中虽然保留了 `distorted_focal_plane` 和畸变多项式接口，但当前 `v2` 主链路并未使用它
    - 仿真端明确加入了 `DVA drift`，且在当前脚本中是实际启用的；`fsglib` 配置中虽有 `enable_dva` 字段，但当前运行链路中尚未实现任何 DVA 修正逻辑
    - 因此当前链路的实际状态是：仿真图像包含 DVA / pointing / thermal 等逐帧运动，而 `fsglib` 主要通过姿态解算去“吸收”这些效应，并未做显式物理修正
    - 当前 `.npz` 和 `run_meta.json` 仍未导出逐帧星位真值，因此静态 `stars.ecsv` 不能直接用于剥离 DVA / jitter / pointing 引起的逐帧真实星位漂移
  - `roll` 误差专项首轮结论：
    - `batch2` 的 matched truth 像面误差整体明显放大，径向 / 切向 RMS 均约 `1.6 pix`，因此它更像是“整体星点定位质量差”主导
    - `batch3` 的 matched truth 像面误差场与 `batch0/1` 更接近（径向 / 切向 RMS 约 `0.58 pix`），但 `roll` 仍明显偏大，说明它不只是“切向误差更大”，还可能存在更微弱但有组织的姿态分量偏差
    - 4 个 batch 的 matched 星场几何覆盖（方位覆盖 / 二阶矩条件数）整体相近，`batch3` 并未表现出非常极端的几何退化，因此当前不优先怀疑“星场构型太差”
  - 已新增两份工程资产：
    - `docs/导星原理.md`：维护导星理论链条与当前代码实现的对照说明
    - `scripts/analyze_v2_biases.py`：复现当前 `v2` 首帧 bias、径向/切向误差、相似变换和短序列汇总的分析脚本
- `2026-03-23`
  - 已核查 `Photosim6ft/centroid_bias_audit_single_star.ipynb` 与 `fsglib` 当前 bias correction 链路的一致性
  - 结论：`fsglib` 代码中的 bias 修正方向与 notebook 是一致的，二者都使用 `corrected = measured - predicted_bias`
  - notebook 中 `<0.001"` 量级的结果成立，但其验证条件非常理想化：
    - 单星
    - 纯静态 PSF
    - 无背景
    - 无噪声
    - 无 jitter / DVA / thermal / PRV
    - 小图直接构造 `PreprocessedFrame`
  - 当前 `fsglib` 运行时条件与 notebook 不同，尤其是使用整帧预处理、阈值分割后的连通域质心、真实多星背景与噪声，因此 notebook 的误差上界不能直接等价为运行时误差
  - 当前发现的关键问题并不是“bias 修正链路没接上”，而是“bias 表和当前仿真所用 PSF 模型不匹配”
  - 证据：
    - bias 表 `configs/bias_profiles/photsim6ft_d280_focus_field12_purepsf_31pix.json` 的元数据记录 `selected_field_id = 6`
    - `configs/bias_correction.yaml` 当前把该表绑定到 `psf.active_model_key = photsim6ft_d280_focus_field12`
    - 但当前 `Photosim6ft/et_sim_100_det.py` 在生成星场时，真正写入 `star_data["field_id"] / stars.catalog["Field ID"]` 的仍是 `0`
    - `photsim6ft/ray_cluster.py` 中图像生成实际按 `star_data["field_id"]` 选 PSF model，因此当前运行图像更可能是按 `field_id = 0` 生成，而不是按 bias 表标定时的 `field_id = 6`
  - 在当前 `v3_v1_noise_psf_20f` 首帧上做了数值核查：
    - 原始质心均值误差约 `(-0.311, -0.028) pix`
    - bias 表预测均值约 `(+0.334, +0.058) pix`
    - 当前 `fsglib` 实现按 notebook 同口径执行减法后，误差会恶化到约 `(-0.645, -0.087) pix`
    - 如果反向加上这张表的 bias，首帧平均径向误差反而会从 `0.348 pix` 降到 `0.171 pix`
  - 这说明：当前表的“量级”是相关的，但“方向/相位”与运行时误差场相反；最可能的根因是 PSF field id 不一致，也不排除叠加了坐标轴镜像/口径翻转
  - 下一步建议：
    - 先为当前真实使用的 PSF field id 重新标定 bias 表
    - 或先把仿真端强制切到与 bias 表一致的 field id 后再复核
    - 在此之前，不应再用这张 `field12` 表直接判断当前 `fsglib` 的质心 bias 修正能力
- `2026-03-27`
  - 已把 guide 首帧几何主链切换为 exact `et_focalplane`，不再用 `body_model_proxy` 生成观测 `los_body`
  - 已确认 `et_focalplane.query_detector_sources()` 与 `stars.ecsv["Detector Xpix/Ypix"]` 一致到 `1e-12 pix` 量级
  - 已确认 `npz["truth_x/y_detector_pix"]` 相比 raw detector 坐标多出一个稳定常量偏移：
    - `dx ≈ +0.1326333675 pix`
    - `dy ≈ -0.0500387349 pix`
    - 径向 `≈ 0.1417585453 pix`
  - 该偏移的来源已定位到 `photsim6ft` 仪器层的 telescope FOV offset：
    - `instrumentation.py` 会在 `Telescope FOV Max Offset > 0` 时随机生成 `fov_xy_offset_pix`
    - `configurator.py` 默认该参数是 `1.0 pix`
    - 当前 guide 仿真脚本没有显式把它置零
    - `frame_truth.py` 会把该 offset 加进 NPZ detector truth
    - 但当前 `run_meta.json` 没有显式记录这一项
  - 已建立新的 guide 评估口径：
    - `current_to_frame_truth` 作为首选姿态精度指标
    - `frame_truth_to_nominal_body` 作为 simulator 相对名义 geometry 的固定偏移指标
    - 旧的 `current_to_oracle` 不再作为首选主指标
  - 当前首帧 exact 结果：
    - 理想 `0.065 pix` 质心实验：`current_to_frame_truth ≈ 0.02848"`，`frame_truth_to_nominal_body ≈ 0.43284"`
    - 真实质心实验：`current_to_frame_truth ≈ 0.51533"`，`frame_truth_to_nominal_body ≈ 0.43177"`
  - 当前判断：
    - 几何主链已经基本打通
    - 对“实际仿真帧真值”的姿态误差，当前主瓶颈是质心提取
  - 已新增 / 更新两份 review notebook：
    - [guide_first_frame_review.ipynb](/home/cxgao/ET/FSG/fsglib/guide_first_frame_review.ipynb)
    - [guide_first_frame_real_centroid_review.ipynb](/home/cxgao/ET/FSG/fsglib/guide_first_frame_real_centroid_review.ipynb)
