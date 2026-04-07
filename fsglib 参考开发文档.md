# `fsglib` 参考开发文档

## 1. 文档目的

本文档用于指导 ET 导星算法的 Python 验证版算法库 `fsglib` 的设计与开发。

供项目组成员作为统一的算法实现参考，明确各模块的输入、输出、数据结构、处理流程、边界条件、测试方法和待定项。

本文档明确聚焦于 **PC 端算法验证实现**，不讨论 FPGA/DSP 寄存器、通信驱动、裸机调度、板级资源映射等实现问题。PC 版实现的目标是快速建立一套可跑通、可测量、可替换、可迭代的算法链路，为后续板上工程化提供依据。

---

## 2. 背景与设计依据

现有系统设计中，导星软件分为 FPGA 与 DSP 两部分：FPGA 侧负责图像预处理和星点提取，DSP 侧负责星图识别、匹配以及姿态解算。详细设计报告与方案总结中都明确给出了这一功能分工，其中 FPGA 图像预处理包含 FPN 校正、暗电平修正等，DSP 则根据匹配星对方向矢量完成姿态求解。

DSP 任务书进一步明确了两种核心工作模式：
一是“初始姿态建立模式”，在无可靠先验姿态或姿态丢失时，基于星敏感器提供的粗姿态和局部子天区匹配建立初始姿态；
二是“星跟踪模式”，基于上一帧姿态结果预测当前帧导航星投影位置，在预测窗口内进行快速匹配。姿态确定算法主线为 QUEST，且需要支持光行差、一阶相对论修正等观测模型修正。

导星仪指标方面，现有材料给出的系统级目标是：凌星方向平动测量精度优于 0.02 角秒、旋向优于 0.05 角秒，微引力方向平动优于 0.03 角秒、旋向优于 0.8 角秒，刷新频率不低于 4 Hz。每套导星由 4 片探测器构成。

进一步明确到当前项目验收口径，需采用以下精度指标：

- 凌星相机单帧精度：`<= 0.02"`（1σ，非绕光轴），`<= 0.05"`（1σ，绕光轴）
- 凌星相机 1 min 多帧叠加精度：`<= 0.003"`（1σ，非绕光轴），`<= 0.01"`（1σ，绕光轴）
- 微引力导星相机单帧精度：`<= 0.03"`（1σ，非绕光轴），`<= 0.8"`（1σ，绕光轴）
- 微引力导星相机 10 min 多帧叠加精度：`<= 0.003"`（1σ，非绕光轴），`<= 0.1"`（1σ，绕光轴）

当前 `fsglib` 的第一优先验证对象是 **凌星相机精度闭环**；微引力导星相机指标应保留在同一套评估框架下，但暂不作为第一阶段主验收项。

性能验证方案中还给出了一个很关键的验证思路：导星单像元角分辨率约 3″，要求单星定位精度不低于 1/5 像素，并通过 4 片探测器总计识别约 400 颗恒星来支撑姿态精度达到 ≤30 mas（1σ）量级。

因此，`fsglib` 的首版必须同时回答两个问题：

1. 这条 5 模块链路是否在仿真星图上可闭环跑通；
2. 这条链路在给定光学/探测器/星场条件下，能否逼近或支撑既定的导星精度目标。

---

## 3. `fsglib` 的开发目标与非目标

## 3.1 开发目标

`fsglib` 第一阶段的目标不是“工程最终版”，而是“算法验证版”。它必须满足以下要求：

- 能读取仿真星图输入；
- 能在统一的 4 片探测器联合坐标体系下处理数据；
- 能支持初始姿态建立与星跟踪两条逻辑链路；
- 能对 5 个核心模块分别输出中间结果；
- 能输出统一的性能评估指标；
- 能支持 1–3 片探测器失效情形下的降级解算；
- 算法实现以 `Python + NumPy + SciPy` 为主，不引入重型依赖。

## 3.2 非目标

第一阶段不追求：

- 板上实时最优实现；
- FPGA 流水/并行结构复现；
- DSP 定点化；
- 上位机/下位机控制协议；
- 在轨遥测遥控；
- 最终冻结的星库格式；
- 最终冻结的光学标定参数格式。

---

## 4. 总体算法链路

`fsglib` 的总体链路应严格对应 5 个算法模块：

1. 图像预处理
2. 星点提取
3. 星图识别
4. 星历查找
5. 姿态解算

从功能上看，这正是将原本板上 FPGA/DSP 的处理主链路抽取为 PC 版统一库。现有设计中，FPGA 负责对图像做预处理并提取星点，DSP 负责根据观测星与导航星匹配完成姿态确定；而在 Python 验证版中，这 5 个模块全部在本机执行，只是逻辑分工仍保持与板级设计一致。

建议定义如下抽象处理流程：

```text
仿真星图 NPZ
  -> detector frame decode
  -> preprocess
  -> source extraction
  -> candidate star vectors
  -> (initial acquisition | tracking)
  -> catalog subregion / predicted stars
  -> association
  -> attitude solve
  -> metrics / logs / debug artifacts
```

---

## 5. 输入数据与统一数据模型

## 5.1 当前已知输入

当前仿真星图输入为 `.npz`，其至少包含如下字段：

- `images`: `(1, 1, H, W)`，`float32`
- `variant_ids`
- `coadd_start`
- `coadd_stop`
- `time_s`
- `cadence_s`
- `unit`

这说明第一阶段应把一帧仿真图像抽象为“单时刻、单探测器或单图块”的图像输入对象，同时保留时间标签、叠加起止和采样节拍信息。

## 5.2 必须建立的统一数据对象

建议在 `fsglib/common/types.py` 中尽早固定以下核心数据结构。

### 5.2.1 `RawFrame`

表示单探测器原始输入帧。

```python
@dataclass
class RawFrame:
    detector_id: int
    image: np.ndarray          # (H, W), float32/float64
    time_s: float
    cadence_s: float
    coadd_start: int | None
    coadd_stop: int | None
    unit: str | None
    variant_id: int | None
    meta: dict
```

### 5.2.2 `PreprocessedFrame`

表示预处理后的图像及其质量掩膜。

```python
@dataclass
class PreprocessedFrame:
    detector_id: int
    image: np.ndarray
    background: np.ndarray | float
    noise_map: np.ndarray | float
    valid_mask: np.ndarray
    preprocess_meta: dict
```

### 5.2.3 `StarCandidate`

表示星点提取输出的单个候选星。

```python
@dataclass
class StarCandidate:
    detector_id: int
    source_id: int
    x: float
    y: float
    flux: float
    peak: float
    area: int
    snr: float
    bbox: tuple[int, int, int, int]
    shape: dict
    flags: dict
```

### 5.2.4 `ObservedStar`

表示已经从像面坐标转换为观测视线方向的星。

```python
@dataclass
class ObservedStar:
    detector_id: int
    source_id: int
    x: float
    y: float
    los_body: np.ndarray       # shape (3,)
    flux: float
    snr: float
    weight: float
    flags: dict
```

### 5.2.5 `MatchedStar`

```python
@dataclass
class MatchedStar:
    detector_id: int
    source_id: int
    catalog_id: int
    los_body: np.ndarray
    los_inertial: np.ndarray
    residual_arcsec: float
    weight: float
    match_score: float
    flags: dict
```

### 5.2.6 `AttitudeSolution`

```python
@dataclass
class AttitudeSolution:
    q_ib: np.ndarray           # quaternion, inertial -> body
    c_ib: np.ndarray           # DCM, optional cache
    euler_zyx: np.ndarray | None
    valid: bool
    mode: str                  # init / tracking / degraded / lost
    num_matched: int
    residual_rms_arcsec: float
    residual_max_arcsec: float
    quality: dict
```

---

## 6. 坐标体系与 4 片探测器联合处理原则

这是整个 `fsglib` 最需要从一开始就定死的部分。否则后面所有模块都会反复返工。

## 6.1 必须区分的坐标系

建议至少定义以下坐标系：

1. 星表惯性系
   初期统一采用 J2000/ICRS 等效惯性系。
2. 望远镜本体系 / 导星仪本体系
   这是姿态解算输出所在的主体系。
3. 单探测器像平面坐标系
   每片探测器自身的 `(x, y)` 像素坐标。
4. 单探测器视线坐标系
   从像面坐标通过内参、畸变修正映射到单位方向矢量。
5. 四片联合焦面坐标系
   用于统一几何建模、联合统计和多片失效降级。

## 6.2 联合处理原则

由于现有设计中每套导星由 4 台探测器构成，且其组合用于高精度姿态解算，因此 Python 版不能把 4 片只当作“后处理时拼一拼的四张图”，而应从数据模型层面支持 4 片联合观测。方案总结明确提到：为提升姿态准确度，姿态解算模块将全部 4 片探测器对应的观测恒星矢量组合为统一观测矩阵，再采用 QUEST 解算。

因此建议遵循：

- 每片探测器独立完成预处理和星点提取；
- 每片探测器独立完成像面到视线方向映射；
- 所有有效 `ObservedStar` 在进入匹配和姿态解算前合并；
- 合并时保留 `detector_id`，绝不丢失来源信息；
- 对 1–3 片失效场景，只是“观测向量子集减少”，而不是切换另一套算法。

---

## 7. 建议目录结构

你前面认可的目录结构是正确的，我建议再细化一层：

```text
fsglib/
├─ __init__.py
├─ common/
│  ├─ types.py
│  ├─ enums.py
│  ├─ constants.py
│  ├─ geometry.py
│  ├─ coords.py
│  ├─ quaternions.py
│  ├─ io.py
│  ├─ masks.py
│  └─ metrics.py
├─ preprocess/
│  ├─ __init__.py
│  ├─ pipeline.py
│  ├─ bias_dark.py
│  ├─ flat_fpn.py
│  ├─ background.py
│  ├─ denoise.py
│  └─ quality.py
├─ extract/
│  ├─ __init__.py
│  ├─ threshold.py
│  ├─ labeling.py
│  ├─ moments.py
│  ├─ centroid.py
│  ├─ filters.py
│  └─ pipeline.py
├─ match/
│  ├─ __init__.py
│  ├─ init_acq.py
│  ├─ local_region.py
│  ├─ geometric_features.py
│  ├─ nearest_neighbor.py
│  ├─ validation.py
│  └─ tracking.py
├─ ephemeris/
│  ├─ __init__.py
│  ├─ catalog.py
│  ├─ proper_motion.py
│  ├─ dva.py
│  ├─ relativity.py
│  ├─ region_select.py
│  └─ predictor.py
├─ attitude/
│  ├─ __init__.py
│  ├─ quest.py
│  ├─ weights.py
│  ├─ outlier.py
│  ├─ residuals.py
│  └─ solver.py
├─ configs/
│  ├─ base.yaml
│  ├─ transit.yaml
│  ├─ microlensing.yaml
│  └─ detector_layout.yaml
├─ tests/
│  ├─ test_preprocess.py
│  ├─ test_extract.py
│  ├─ test_match.py
│  ├─ test_ephemeris.py
│  ├─ test_attitude.py
│  └─ test_end2end.py
└─ examples/
   ├─ run_single_frame.py
   ├─ run_sequence_tracking.py
   └─ evaluate_sim_dataset.py
```

---

## 8. 配置系统设计原则

由于凌星与微引力共用一套算法框架，但参数不同，因此必须坚持“同一代码、参数切换”的设计，而不是复制两套代码。

建议把配置分为 5 层：

1. 探测器参数
   像元尺寸、尺寸、读出方向、坏点图、增益、饱和阈值等
2. 光学参数
   焦距、主点、畸变模型、每片探测器安装矩阵、视场边界
3. 提取参数
   背景估计窗口、阈值倍数、最小面积、最大面积、形态学筛选门限
4. 匹配参数
   粗姿态容差、预测窗口半径、近邻半径、几何一致性阈值、重捕获阈值
5. 姿态解算参数
   权重模型、异常剔除门限、最小匹配星数、解有效性判据

---

# 9. 模块 1：图像预处理

## 9.1 模块目标

图像预处理模块的任务不是“把图像变好看”，而是把原始仿真图像转换为 **适合星点提取的数值场**。
它的核心目标只有三个：

1. 降低背景和固定模式误差对星点检测的干扰；
2. 让同一颗星在不同帧、不同探测器上的响应具备尽可能一致的统计意义；
3. 保留质心定位所需的亚像素结构，不要把 PSF 洗平。

在现有设计材料中，FPGA 侧图像预处理明确包括 FPN 校正、暗电平修正等操作。

## 9.2 输入

输入为 `RawFrame`，至少包含：

- `image`
- `detector_id`
- `time_s`
- 相关观测元数据

此外，预处理模块需要引用外部校准与配置数据：

- bias / dark 模型
- FPN / flat / PRNU 模型
- 坏点图
- 饱和像素阈值
- 可选的背景模型参数
- 可选的卷帘时序参数

## 9.3 输出

输出为 `PreprocessedFrame`：

- `image`: 预处理后图像
- `background`: 背景估计
- `noise_map`: 噪声估计
- `valid_mask`: 有效像素掩膜
- `preprocess_meta`: 记录各步骤的实际参数与统计量

## 9.4 建议处理流程

推荐第一阶段按如下顺序实现：

### 第 1 步：基础读入与维度标准化

把 `(1, 1, H, W)` 统一转为 `(H, W)`。
同时记录原始 dtype、最值、NaN/Inf 状态。

### 第 2 步：非法值处理

将 NaN/Inf 标记到 `valid_mask=False`，数值上可置 0 或局部插值，但必须保留掩膜。

### 第 3 步：暗电平 / bias 校正

如果仿真图像已经是物理单位图，不一定需要减 bias；但接口上必须保留该步骤。
因为将来真实数据或更接近真实电子学链路的仿真，都会需要这一层。

### 第 4 步：FPN / 平场一致性修正

当前详细设计中已明确提到 FPN 校正，因此接口必须预留。

第一阶段可以支持三种模式：

- `none`
- `global_gain_offset`
- `pixelwise_flat`

### 第 5 步：坏点 / 饱和像素屏蔽

对以下像素设为无效：

- 明确坏点图中的坏点
- 超过饱和阈值的像素
- 负值异常到不可物理解释的像素（可选）

### 第 6 步：背景估计

推荐优先使用稳健背景估计，而不是简单全图均值。建议：

- sigma-clipped median
- 分块 background map + 双线性回插
- 或大核平滑背景，但避免吞掉大尺度星场结构

### 第 7 步：背景扣除

得到用于检测与质心计算的“净信号图”。

### 第 8 步：噪声图估计

建议噪声图至少能表示为：

[
\sigma(x,y)=\sqrt{\sigma_{read}^2+\sigma_{bg}^2+\max(I(x,y),0)/g}
]

即使当前只是仿真图像，也应保留这一抽象。后续阈值、SNR、权重都要依赖它。

### 第 9 步：可选轻度去噪

第一阶段只允许使用非常保守的去噪，如：

- 3x3 中值滤波，仅用于坏点邻域
- 非强制的小尺度高斯平滑，且默认关闭

原则是：**不得为了“检测更稳”而破坏质心亚像素结构**。

## 9.5 推荐接口

```python
def preprocess_frame(
    raw: RawFrame,
    calib: dict,
    cfg: dict
) -> PreprocessedFrame:
    ...
```

## 9.6 关键设计约束

### 约束 1：预处理必须“可逆可审计”

每一步都要在 `preprocess_meta` 中记录：

- 是否启用
- 参数值
- 处理前后均值/标准差
- 屏蔽像素数量

### 约束 2：预处理不得隐藏问题

例如不应把饱和星点通过插值“修复”成正常星点参与姿态解算。
这类目标应在后续提取或匹配环节降权或剔除。

### 约束 3：预处理必须支持模式切换

凌星与微引力应共用相同接口，但允许：

- 不同背景估计窗口
- 不同阈值
- 不同平场/噪声参数

## 9.7 边界条件

必须单独测试：

- 全黑帧 / 近全黑帧
- 极亮星导致局部饱和
- 多颗相邻亮星
- 存在坏点簇
- 背景梯度明显
- 四片探测器统计特性不同

## 9.8 测试方法

### 单元测试

输入人工构造图像，验证：

- bias/FPN 修正是否符合预期
- 坏点掩膜是否正确
- 背景估计是否稳定

### 仿真测试

基于真实仿真星图，检查：

- 预处理前后背景 RMS
- 检测前 SNR 变化
- 质心偏差是否变差

### 回归测试

固定样本集，锁定一组统计输出：

- `num_invalid_pixels`
- `background_median`
- `noise_median`
- `image_std_after_subtraction`

## 9.9 待定项

这里是你需要继续向光学/探测器/仿真侧确认的信息：

1. 仿真图像是否已经包含 bias、dark、PRNU、读出噪声
2. 仿真图像单位到底是 e-、ADU、归一化通量还是其他
3. 是否已经注入坏点/热像元/列缺陷
4. 是否包含卷帘读出的时序差异
5. 是否包含暗场不均匀性与像元响应不均匀性
6. 仿真图像中的 PSF 是否已经叠加探测器扩散与采样效应

---

# 10. 模块 2：星点提取

## 10.1 模块目标

星点提取模块的任务是从预处理后图像中，得到稳定、可解释、可筛选的星点候选列表。
方案总结中明确指出该模块包含两步：粗定位与亚像素定位。粗定位本质上是图像分割，目标是把目标星与背景、以及不同星点之间分离；亚像素定位则采用基于灰度的方法实现高于像元尺度的定位精度。

这与您当前确定的路线“阈值分割 + 连通域 + 质心法”完全一致，适合作为第一阶段主线。

## 10.2 输入

输入为 `PreprocessedFrame`：

- 预处理图像
- 背景信息
- 噪声图
- 有效掩膜

## 10.3 输出

输出为 `list[StarCandidate]`。

每个候选星必须至少包含：

- 亚像素质心 `(x, y)`
- 通量 `flux`
- 峰值 `peak`
- SNR
- 连通域面积
- 候选框 `bbox`
- 形状统计量
- 标志位 `flags`

## 10.4 推荐处理流程

### 第 1 步：检测图构建

最简单有效的第一阶段实现是：

[
D = \frac{I_{pre}}{\sigma_{noise}}
]

即用“信号/噪声图”作为检测图，而不是原图直接阈值。

### 第 2 步：阈值分割

推荐采用双阈值机制：

- `seed_threshold`: 高阈值，用于种子像素
- `grow_threshold`: 低阈值，用于区域扩展

这样比单阈值更稳，尤其适合 PSF 翼部较弱时。

### 第 3 步：连通域标记

对检测图中有效区域做 8 邻域连通域标记，获得候选源区域。

### 第 4 步：初筛

剔除明显不是恒星的区域：

- 面积过小：噪声尖峰
- 面积过大：背景波动、鬼像或拖尾
- 峰值过低：低可信度
- 靠边过近：质心不完整
- 形状过扁或过长：可能不是正常星像

### 第 5 步：粗定位

粗定位可用：

- 最大值像素位置
- 连通域几何中心
- 或强度加权整数近似中心

### 第 6 步：亚像素定位

第一阶段推荐实现三种方法，但默认只启用第一种：

1. **灰度加权质心**
2. 高斯二次曲面局部拟合
3. PSF 模板拟合（作为增强版）

其中，第一种最稳健、最容易快速落地。

### 第 7 步：通量与形状统计

建议提取：

- 总通量
- 峰值
- 二阶矩
- 椭率
- 主轴方向
- 半能量半径或近似尺度

这些统计量后续可用于：

- 质量打分
- 权重计算
- 双星/坏源剔除

## 10.5 质心计算建议

第一阶段使用窗口加权质心：

[
x_c = \frac{\sum I_i x_i}{\sum I_i}, \quad
y_c = \frac{\sum I_i y_i}{\sum I_i}
]

但必须明确三个实现细节：

### 细节 1：必须做背景扣除

不能直接用原始灰度做质心，否则背景偏置会把质心往窗口中心拉。

### 细节 2：必须限制积分窗口

不要对整个连通域无限扩展。建议：

- 使用连通域 bbox 外扩固定半径；
- 或使用以峰值点为中心的固定窗口；
- 或使用阈值裁剪后的局部窗口。

### 细节 3：必须支持异常标志

对以下情况打标：

- `saturated`
- `edge_truncated`
- `blended`
- `elongated`
- `low_snr`

这些标志后续不能丢。

## 10.6 推荐接口

```python
def extract_stars(
    frame: PreprocessedFrame,
    cfg: dict
) -> list[StarCandidate]:
    ...
```

以及更细粒度接口：

```python
def detect_sources(image, noise_map, mask, cfg) -> list[np.ndarray]:
    ...

def measure_source(image, segmask, cfg) -> StarCandidate:
    ...
```

## 10.7 关键设计约束

### 约束 1：检测与测量分离

“找到候选源”和“计算精确质心”必须分成两个阶段，否则调试会很痛苦。

### 约束 2：候选源宁可略多，不可过早删光

第一阶段更重要的是召回率，不是最终纯净度。
错误候选可以在匹配和姿态解算前继续筛掉；但漏掉高质量导航星会直接伤害姿态精度。

### 约束 3：必须支持四片探测器独立统计

每片探测器需要单独输出：

- 候选星数
- 可用星数
- 平均 SNR
- 平均质心误差估计

## 10.8 评价指标

星点提取模块至少输出以下指标：

- `num_detected`
- `num_usable`
- `num_saturated`
- `num_edge_truncated`
- `mean_snr`
- `median_snr`
- `centroid_internal_scatter`
- 与仿真真值的匹配召回率（若真值可得）
- 虚警率

## 10.9 与系统精度的关系

验证方案中把单星定位误差拆成 TE、HSFE、LSFE 三类，并给出组合关系：

[
\sigma_s^2 = TE^2 + HSFE^2 + LSFE^2
]

因此，星点提取模块不能只输出“找到多少颗星”，还必须开始为后续精度分解预留统计接口。

建议从第一阶段开始就输出：

- 同一星点重复观测下的质心散布，作为 TE 的近似观测量；
- 不同亚像素相位位置下的系统性偏差，作为 HSFE 的候选观测量；
- 跨视场位置/温度/长期漂移趋势的系统差，作为 LSFE 的候选观测量。

## 10.10 边界条件

必须单独验证：

- 星点重叠 / blend
- 高亮星饱和
- 低 SNR 星
- 靠边星
- 拖尾/轻微非对称 PSF
- 单帧仅少量星点
- 局部背景不均匀

## 10.11 待定项

1. PSF 数据格式是什么，是否可直接做模板拟合
2. 允许的最小 SNR 门限是多少
3. 饱和阈值与线性区上限是否已知
4. 是否已有仿真真值星表可用于直接评估提取误差
5. 是否希望在第一阶段就支持 blend deblending

---

# 11. 模块 3：星图识别

## 11.1 模块目标

星图识别模块的目标不是“做一个通用全天星敏感器识别器”，而是在 **ET 导星场景** 下，利用先验姿态，把搜索空间压缩到局部子天区，然后对观测星图与导航星库做稳定匹配。

现有设计已经把路线定得很清楚：
初始姿态建立模式下，依据星敏感器粗姿态确定潜在子天区，再读取对应导航星进行匹配；
星跟踪模式下，依据上一帧姿态预测当前帧导航星在探测器上的投影位置，在预测窗口内快速匹配。

方案总结和详细设计也明确指出，星图识别采用“局部子天区匹配框架”，并使用“分区参考星图 + 近邻匹配”的方案思想。

## 11.2 模块内部应拆成两个子模式

### 模式 A：初始姿态建立（initial acquisition）

适用于：

- 上电后首次建立指向
- 长时间失锁后重捕获
- 连续失败退回重识别

其输入是：

- 当前帧提取得到的观测星点
- 星敏感器提供的粗姿态
- 安装矩阵
- 当前时刻与光学参数

其输出是：

- 已识别的观测星—导航星对应关系
- 初始姿态估计
- 匹配质量指标

### 模式 B：星跟踪匹配（tracking association）

适用于：

- 已经有上一帧有效姿态结果
- 需要快速更新当前帧匹配

其输入是：

- 上一帧姿态
- 当前时刻
- 导航星历史跟踪表
- 当前帧观测星点

其输出是：

- 更新后的匹配结果
- 新增/丢失星列表
- 当前跟踪质量指标

## 11.3 输入

输入建议为：

```python
@dataclass
class MatchingContext:
    mode: str                     # init | tracking
    time_s: float
    observed_stars: list[ObservedStar]
    prior_attitude: np.ndarray | None
    detector_layout: dict
    optical_model: dict
    matching_cfg: dict
```

## 11.4 输出

输出建议为：

```python
@dataclass
class MatchingResult:
    matched: list[MatchedStar]
    unmatched_observed_ids: list[int]
    unmatched_catalog_ids: list[int]
    mode: str
    success: bool
    score: float
    debug: dict
```

## 11.5 初始姿态建立模式的推荐流程

### 第 1 步：粗姿态转换

根据星敏感器姿态与安装矩阵，得到望远镜/导星仪粗姿态。
方案总结中明确写到：可通过星敏感器当前姿态 `As` 与安装矩阵 `T`，得到望远镜初始姿态 `At = T × As`。

### 第 2 步：潜在子天区快速查找

依据粗姿态、视场边界、4 片探测器布局，从星表中筛出当前潜在可见的导航星。

这里必须注意：
因为是导星，不是全全天搜索，所以 **绝不应默认全星表暴力匹配**。
第一阶段即使性能不是瓶颈，也应按未来工程形态保留“子天区筛选”结构。

### 第 3 步：构造观测几何特征

DSP 任务书中建议使用星间角距、角距比或其他几何不变量，并提到金字塔算法。

第一阶段建议不要一步上完整复杂金字塔库，而采用折中方案：

- 从最亮若干观测星中构造星对/星三角形特征；
- 与子天区候选星构造的参考特征做匹配；
- 得到若干候选对应关系；
- 用姿态解算 + 回投影一致性做最终验证。

### 第 4 步：候选对应关系生成

推荐先实现：

- 角距容差匹配
- 多特征联合评分
- 候选数截断

### 第 5 步：候选验证

候选验证是识别的真正核心。方案总结中明确提到了“金字塔内核验证法”的思想：如果某个匹配成立，用其解出的姿态生成参考星图，则视场内其他观测星也应在较小邻域内找到对应导航星。

这实际上给出了非常好的工程判据：

1. 用少量候选星对先解一个粗姿态；
2. 将子天区导航星投影到 4 片探测器；
3. 在小邻域内寻找更多一致匹配；
4. 若支持匹配数足够多且残差足够小，则判定识别成功。

### 第 6 步：输出初始匹配结果

输出必须包括：

- 匹配星对
- 识别成功标志
- 支持匹配数
- 几何一致性评分
- 候选子天区 ID 或其等价标记

## 11.6 星跟踪模式的推荐流程

### 第 1 步：基于上一帧姿态预测导航星投影

这是跟踪模式的基础，也是与初始识别模式的本质区别。DSP 任务书对此已经明确规定。

### 第 2 步：预测窗口搜索

对每颗已知导航星，在当前帧每片探测器上给出预测位置与搜索半径。

### 第 3 步：近邻匹配

第一阶段推荐：

- 位置近邻为主
- 亮度/SNR 一致性为辅
- 必要时加入局部形状一致性

### 第 4 步：跟踪星表维护

DSP 任务书要求支持：

- 新星进入视场时纳入列表
- 离场或质量下降时剔除
- 匹配星数不足时重捕获或退回初始模式。

因此在 Python 验证版中，必须把“跟踪表维护”作为显式数据结构，而不是临时列表。

建议定义：

```python
@dataclass
class TrackState:
    catalog_id: int
    detector_id: int | None
    last_xy: tuple[float, float] | None
    last_seen_time_s: float
    miss_count: int
    quality_score: float
    active: bool
```

## 11.7 关键设计约束

### 约束 1：初始识别与跟踪匹配是两套逻辑，不要混写

虽然底层可以共享投影、近邻、残差等函数，但主流程必须分开。

### 约束 2：必须有“识别成功判据”

不能因为找到了几对看起来像的星，就直接输出姿态。
必须至少满足：

- 匹配星数超过门限
- 残差 RMS 小于门限
- 匹配分布不退化
- 四元数/姿态更新与先验不矛盾

### 约束 3：必须容忍误检、漏检

DSP 任务书明确要求匹配算法允许存在一定比例的误检或漏检，并通过一致性判据剔除错误匹配。

### 约束 4：匹配模块只负责“关联”，不应承担最终姿态真值判断

它应输出候选与质量信息，最终解是否有效要由姿态解算模块再判。

## 11.8 建议接口

```python
def match_stars(
    ctx: MatchingContext,
    catalog_provider,
    projector,
    cfg: dict
) -> MatchingResult:
    ...
```

子接口建议拆为：

```python
def initial_acquire(...): ...
def predict_catalog_positions(...): ...
def associate_nearest(...): ...
def validate_match_hypothesis(...): ...
def update_track_table(...): ...
```

## 11.9 核心评估指标

星图识别模块至少要输出：

- 初始识别成功率
- 星跟踪保持率
- 平均匹配星数
- 错配率
- 重捕获触发频率
- 连续稳定跟踪帧数
- 单帧匹配耗时

验证方案中明确提出外场测试需要验证星图识别有效性、输出识别恒星数目、以及算法实时性。这些应直接变成 `fsglib` 的标准评估项。

## 11.10 边界条件

必须测试：

- 粗姿态误差偏大
- 可见导航星数偏少
- 提取结果中混入误检源
- 部分探测器无有效星点
- 上一帧姿态漂移较大
- 亮度排序变化
- 局部子天区存在几何相似混淆

## 11.11 待定项

1. 子天区划分的具体策略：HEALPix、经纬网格还是自定义四叉分区
2. 初始识别时采用星对、三角形还是精简金字塔
3. 识别成功的最小匹配星数门限
4. 预测窗口半径是否与姿态协方差联动
5. 亮度是否作为强判据，还是只作为软约束

---

# 12. 第一阶段就必须统一输出的评估指标

虽然姿态精度是最终主指标，但从工程上讲，第一阶段每个模块都必须输出自己的局部指标，否则最后无法定位误差来源。

建议统一输出以下指标树：

## 12.1 预处理层

- 背景均值 / 中位数 / RMS
- 无效像素比例
- 饱和像素比例
- 预处理耗时

## 12.2 星点提取层

- 候选星数
- 可用星数
- 平均 SNR
- 召回率 / 虚警率
- 质心误差分布
- 处理耗时

## 12.3 匹配层

- 初始识别成功率
- 跟踪保持率
- 平均匹配星数
- 错配率
- 重捕获次数
- 匹配耗时

## 12.4 姿态层（下一部分展开）

- 姿态误差三轴分量
- 四元数误差角
- 残差 RMS / Max
- 有效解比例
- 失效场景下降级性能

---

# 13. 必须向光学工程师继续确认的输入清单

你前面问“其他光学数据还需要哪些”，这里我直接给出一份应索取清单。这一段非常关键，因为缺这些，模块 4 和模块 5 很容易写成空架子。

## 13.1 必须要有的几何光学输入

1. 每片探测器到望远镜本体系的安装矩阵
2. 每片探测器的主点 `(cx, cy)`
3. 有效焦距 `f`
4. 像元尺寸
5. 畸变模型形式与参数
   - 径向项
   - 切向项
   - 或直接给像面到视线的查找表/多项式
6. 四片探测器在联合焦面中的相对位置与朝向
7. 每片探测器有效视场边界多边形

## 13.2 强烈建议要有的 PSF 相关输入

1. PSF 数据格式
   - 栅格图
   - 参数化模型
   - 不同视场点的离散库
2. PSF 是否随波长变化
3. PSF 是否随视场位置变化
4. PSF 是否随离焦量变化
5. 是否已有 EE80 / EE90 / FWHM 指标
6. 是否已有亚像素响应或 IPC 模型

## 13.3 与时间/观测有关的输入

1. 卷帘读出时序模型
2. 单帧曝光起止定义
3. `coadd_start/coadd_stop` 的严格物理含义
4. 当前仿真时刻对应的参考历元
5. 是否已包含 DVA、光行差、相对论等效应，还是由算法侧补偿

---

# 14. 面向 Copilot 的编码约束

这一段建议你后面直接放进仓库顶层 `CONTRIBUTING_fsglib.md`。

## 14.1 必须遵守

- 仅使用 `Python + NumPy + SciPy`
- 所有核心函数必须类型注解
- 所有模块必须可单独单元测试
- 所有阈值必须来自配置，不得硬编码在核心流程
- 所有中间结果必须支持保存调试产物
- 所有算法步骤必须区分“主逻辑函数”和“评估/可视化函数”

## 14.2 严禁行为

- 在核心函数中混入绘图逻辑
- 在模块内部读写全局状态
- 在匹配阶段偷偷做姿态有效性兜底
- 在姿态解算阶段偷偷改观测数据
- 使用难以板上迁移的重型依赖
- 为了过样例而写面向单一数据集的特殊分支

## 14.3 推荐编码风格

每个模块都采用：

```python
def run_xxx(input_obj, model, cfg) -> output_obj:
    ...
```

并配套：

```python
def validate_xxx(output_obj, truth, cfg) -> dict:
    ...
```

这样后期无论是做板上迁移，还是做误差归因，都非常清楚。

# 16. 模块 4：星历查找

## 16.1 模块目标

“星历查找”在 `fsglib` 中不能写成一个含糊的大杂烩模块。它的职责应当被严格限定为：

1. 从星表与时间信息出发，生成当前时刻可用于匹配与解算的参考恒星集合；
2. 在初始姿态建立模式下，根据粗姿态和视场约束，筛出潜在子天区导航星；
3. 在星跟踪模式下，根据上一帧姿态结果与时间推进，预测当前帧导航星在 4 片探测器上的理论投影位置；
4. 对参考星方向矢量施加必要的时间推进与观测模型修正，为后续匹配和姿态解算提供“正确的参考量”。

DSP 任务书已经把这个模块的外部行为说得很清楚：初始模式要依据星敏感器粗姿态确定潜在子天区并读取对应导航星；星跟踪模式则要根据上一帧姿态结果和系统角速度假设，预测当前帧导航星在探测器上的投影位置；姿态解算前还需要根据需要对星点方向做光行差、一阶相对论修正。

所以，这个模块本质上不是“天体力学模块”，而是“参考星生成与投影预测模块”。

## 16.2 模块边界

这个模块只负责三类输出：

- 参考星在某个统一惯性系中的方向矢量；
- 参考星经过系统几何与姿态投影后，在各探测器上的理论像面位置；
- 供匹配与姿态解算使用的参考属性，如星等、权重初值、是否可见、是否越界。

它不负责：

- 直接判定观测星和参考星是否匹配；
- 直接给出最终姿态；
- 直接做质心定位误差补偿；
- 直接替代光学标定。

## 16.3 两个工作子模式

### 16.3.1 初始姿态建立模式下的星历查找

在该模式下，输入核心是：

- 当前时刻 `time_s`
- 星敏感器先验姿态
- 星敏到导星仪的安装矩阵
- 4 片探测器的视场边界与安装参数

输出是“当前潜在可见导航星列表”。

这是一个“粗筛选 + 精投影”的过程。DSP 任务书对此要求非常明确：根据星敏感器提供的粗略姿态信息，确定导星探测器观测恒星所在潜在子天区，读取对应子天区导航星进行匹配。

### 16.3.2 星跟踪模式下的星历查找

在该模式下，输入核心变成：

- 上一帧姿态解算结果
- 当前时刻与上一帧时刻
- 系统角速度假设
- 当前有效跟踪星表

输出不再是“大范围候选星”，而是“每颗已知导航星在当前帧上的预测投影位置”。

DSP 任务书同样明确要求：根据上一帧姿态解算结果及系统角速度假设，预测当前帧导航星在探测器上的投影位置，并在预测位置邻域内进行匹配。

这意味着 `fsglib` 的模块实现必须把“初始筛选”和“跟踪预测”写成两条显式分支，而不是一个函数里用很多 `if` 搅在一起。

## 16.4 推荐输入与输出数据结构

建议在 `fsglib/ephemeris/types.py` 中定义以下对象。

```python
@dataclass
class CatalogStar:
    catalog_id: int
    ra_deg: float
    dec_deg: float
    pm_ra_mas_per_yr: float | None
    pm_dec_mas_per_yr: float | None
    parallax_mas: float | None
    rv_km_s: float | None
    mag_g: float | None
    color_bp_rp: float | None
    meta: dict
@dataclass
class ReferenceStar:
    catalog_id: int
    time_s: float
    los_inertial: np.ndarray      # (3,)
    mag_g: float | None
    detector_ids_visible: list[int]
    predicted_xy: dict[int, tuple[float, float]]
    predicted_valid: dict[int, bool]
    weight_hint: float
    meta: dict
@dataclass
class EphemerisContext:
    mode: str                     # init | tracking
    time_s: float
    prior_attitude_q: np.ndarray | None
    angular_rate_body: np.ndarray | None
    detector_model: dict
    optical_model: dict
    catalog_cfg: dict
    correction_cfg: dict
```

## 16.5 建议拆分为“离线建库”和“在线查询”两层

这是非常关键的工程设计点。

### 16.5.1 离线建库层

由于你已经确定星表采用 Gaia，但当前尚未下载，所以 `fsglib` 第一阶段应当把星表层设计成“接口稳定，后端可替换”的形式。

离线建库至少要做这些事：

1. 从 Gaia 原始表中筛出任务需要的星等范围；
2. 为每颗星保存最基本的天体测量量：
   - RA/Dec
   - proper motion
   - parallax（如可得）
   - radial velocity（如可得）
   - magnitude
3. 建立空间索引，以支持“按粗姿态和视场快速查找子天区”；
4. 预留星表裁剪缓存，避免在线每次查询都扫全表。

推荐第一版就把它抽象成：

```python
class CatalogProvider:
    def query_region(self, boresight, radius_deg, mag_limit) -> list[CatalogStar]:
        ...
```

### 16.5.2 在线查询层

在线层只做本次时间推进和本次视场投影。
不要在在线层混入任何“读 Gaia 大文件”的行为。

## 16.6 时间推进与参考方向矢量生成

现有性能验证方案对地面外场测试给出了比较完整的时间推进链路：从 J2000.0 时刻的赤经、赤纬和视运动参数出发，先得到时刻 (T) 的 J2000 方向矢量，再进一步考虑岁差、章动，并在最终用 QUEST 与地球自转后的参考矢量做比较。

对 `fsglib` 来说，这些材料有两个意义：

第一，它们证明“时间推进和参考矢量生成”本身就是导星验证链路的一部分，不能偷懒写成固定星表坐标。

第二，地面外场测试中的完整地固系转换流程，不应原封不动照搬到你的“仿真星图 PC 验证”中。因为仿真星图很可能已经定义在某个惯性参考框架里，或者由仿真器直接给出了视场内恒星像面结果。此时更合理的做法是把时间推进拆成若干可选修正项，按仿真数据真实生成链路来启用。

因此，建议 `fsglib` 中将参考方向生成分成 4 层：

1. `base_astrometry`：J2000 / ICRS 基础星位
2. `proper_motion_update`：视运动推进
3. `astrometric_frame_update`：岁差、章动等参考系更新
4. `observation_corrections`：DVA、光行差、一阶相对论等观测修正

## 16.7 第一阶段建议的修正层级

考虑到当前任务是“算法验证版”，我建议把修正分成三挡，不要一开始就全部强绑死。

### 档位 A：最小可跑通版本

- RA/Dec
- proper motion
- 统一惯性系方向矢量
- 不做章动/岁差
- 不做 DVA
- 不做相对论修正

适用于最初把流程跑通。

### 档位 B：标准验证版本

- RA/Dec
- proper motion
- 统一惯性系方向矢量
- 可选岁差/章动
- 可选光行差
- 可选 DVA

这应该是你们真正要做的主版本。

### 档位 C：扩展精化版本

- RA/Dec
- proper motion
- parallax
- radial velocity
- 岁差/章动
- DVA
- 一阶相对论修正
- 更严格的时间尺度处理

DSP 任务书已经明确要求需要对星点方向进行光行差、一阶相对论等修正，性能验证方案也明确用了从 J2000 到观测时刻的坐标推进链路，因此这部分必须在架构上预留，而不是以后重写。

## 16.8 DVA 与相对论修正的工程建议

你已经明确要求把 DVA 和相对论效应纳入考虑。这个决定是对的，因为这两项如果后面才补，会直接打乱：

- 星表参考方向；
- 像面预测位置；
- 匹配窗口大小；
- 残差统计口径。

但从实现次序上看，我建议不要把 DVA 和相对论写死在姿态解算模块里，而是放在 `ephemeris/corrections.py` 中，作为“参考矢量修正器”。

原因很简单：
DVA 和相对论修正影响的是“参考星真实应在何处”，而不是“如何从已匹配星对求姿态”。

推荐接口如下：

```python
def apply_observation_corrections(
    los_inertial: np.ndarray,
    observer_state: dict,
    cfg: dict
) -> np.ndarray:
    ...
```

这样做的好处是：

- 初始识别模式和跟踪模式共用一套修正逻辑；
- 可以独立开关和做误差对比实验；
- 将来板上化时更容易拆分。

## 16.9 从参考矢量到 4 片探测器预测像点

这个步骤是星历查找和匹配之间的桥梁。

建议统一流程是：

1. 惯性系参考矢量 `los_inertial`
2. 由当前先验姿态变换到导星仪本体系 `los_body`
3. 根据每片探测器安装矩阵变换到单片探测器坐标系
4. 通过光学投影模型得到理想像面坐标
5. 通过畸变模型得到实际预测像面坐标
6. 判断是否落入该片有效视场边界

这里最重要的是：
不要用一个“全焦面简化二维拼接坐标”把光学过程糊掉。
四片探测器虽然最终联合用于姿态解算，但每片的安装误差、主点、畸变、可见边界都应该独立建模。性能验证方案也明确要求焦距、主点、畸变等系统参数需要标定并完成畸变校正。

## 16.10 推荐接口

```python
def build_reference_stars(
    ctx: EphemerisContext,
    catalog_provider,
    cfg: dict
) -> list[ReferenceStar]:
    ...
def predict_detector_positions(
    ref_stars: list[ReferenceStar],
    attitude_q: np.ndarray,
    detector_model: dict,
    optical_model: dict,
    cfg: dict
) -> list[ReferenceStar]:
    ...
```

## 16.11 评价指标

星历查找模块不直接给姿态，但它必须输出以下质量指标：

- `num_catalog_region_candidates`
- `num_predicted_visible_total`
- 每片探测器 `num_predicted_visible_by_detector`
- 预测位置越界比例
- 参考星亮度分布
- 理论参考星与观测星的最近邻平均距离
- 时间推进耗时
- 投影耗时

## 16.12 待定项

这一部分你后续还需要继续问光学和仿真侧：

1. 仿真星图对应的参考坐标系到底是什么
2. 仿真器是否已注入岁差、章动、DVA、光行差、相对论修正
3. 四片探测器边缘给出的 `l,b` 视场点到底对应哪一层坐标定义
4. 是否能提供每片探测器完整边界点列，而不是只给边缘若干点
5. 是否有每片探测器独立的畸变模型或查找表
6. 仿真星图是否有恒星真值表，便于直接验证预测位置误差

---

# 17. 模块 5：姿态解算

## 17.1 模块目标

姿态解算模块是整个 `fsglib` 的核心闭环输出模块。
它的任务是：对已经完成匹配的观测星—参考星对，构造观测矢量和参考矢量，采用加权姿态确定方法解出当前姿态四元数，并输出残差、质量指标和有效性判据。

DSP 任务书对此规定得非常明确：姿态解算应基于已匹配星点完成高精度姿态产品输出；观测模型可根据需要加入光行差和一阶相对论修正；姿态确定算法采用 QUEST；星点权重可与 SNR 或测量不确定度相关；最终输出姿态四元数、状态信息、解算残差和质量指标。

同时，DSP 设计约束还明确要求：内部主变量必须采用四元数，不能把欧拉角作为内部主变量；QUEST 解算必须保证数值稳定性，在几何退化情况下不能输出非物理解；四元数必须归一化。

## 17.2 输入

推荐输入对象如下：

```python
@dataclass
class AttitudeSolveInput:
    time_s: float
    matched_stars: list[MatchedStar]
    prior_q_ib: np.ndarray | None
    mode: str                     # init | tracking
    solver_cfg: dict
```

其中 `matched_stars` 每个元素应至少包含：

- 观测视线矢量 `los_body`
- 参考视线矢量 `los_inertial`
- 星点质量信息
- 探测器来源
- 匹配分数
- 预测残差或匹配残差

## 17.3 输出

输出建议仍然使用上一部分定义的 `AttitudeSolution`，但建议扩展：

```python
@dataclass
class AttitudeQuality:
    num_input: int
    num_used: int
    num_rejected: int
    residual_rms_arcsec: float
    residual_max_arcsec: float
    cond_score: float
    geometry_score: float
    degraded: bool
    reject_reason_counts: dict
```

## 17.4 数学主线

方案总结和详细设计材料里都已经把主线定成了“多星矢量 + QUEST”。并且明确提到：全部 4 片探测器对应的观测恒星矢量应组合为统一观测矩阵，同样将其对应导航星天球位置矢量组合为导航星矩阵，再采用 QUEST 解算姿态矩阵。

因此第一阶段不建议再摇摆去试很多核心姿态算法。
主线就定为：

1. 构造观测矢量集合 ( \mathbf{w}_i )
2. 构造参考矢量集合 ( \mathbf{v}_i )
3. 计算权重 ( \lambda_i )
4. 加权 QUEST 求四元数
5. 残差评估
6. 异常点剔除后重解
7. 输出最终姿态与质量标记

## 17.5 最低观测星数的工程判据

方案总结里提到，从纯几何上讲，两个非平行观测矢量就足以确定三轴姿态。

但这只是“数学可解”的下限，不是“工程可用”的下限。
在 `fsglib` 中，建议分开定义两个门限：

- `min_stars_mathematical = 2`
- `min_stars_operational = 4 or 5`

也就是说：

- 小于 2：绝对不可解；
- 等于 2 或 3：理论可解，但默认标记为高风险/降级；
- 大于等于 4 或 5：作为正常工程解算的最低门限。

这里的 4 或 5 不是文档硬要求，而是我给你的实现建议。因为在实际工程里还要承受：

- 质心误差
- 错配
- 局部几何退化
- 多探测器非均匀分布
- 单颗星异常

## 17.6 权重设计

DSP 任务书已允许星点权重与 SNR 或测量不确定度相关。

因此在 `fsglib` 中，建议权重函数不要写死，而是支持三类模式。

### 模式 A：按测量方差加权

[
\lambda_i \propto \frac{1}{\sigma_i^2}
]

其中 ( \sigma_i ) 来自质心误差估计、像面到视线映射误差估计或经验模型。

### 模式 B：按 SNR 加权

[
\lambda_i \propto \mathrm{SNR}_i^\gamma
]

其中 ( \gamma ) 通常取 1 或 2。

### 模式 C：混合权重

[
\lambda_i \propto \frac{\mathrm{score}_i}{\sigma_i^2}
]

这里 `score_i` 可以综合：

- 匹配分数
- 残差一致性
- 形状质量
- 是否边缘截断
- 是否饱和

第一阶段建议默认使用“测量方差优先 + SNR 修正”的混合权重。

## 17.7 残差定义与解后验证

解算不是求出一个四元数就结束。
还必须明确“如何判断这帧姿态是否可信”。

推荐解后至少计算两类残差：

### 17.7.1 角残差

把参考矢量旋转到观测系后，与观测矢量比较夹角：

[
r_i = \arccos \left( \mathbf{w}_i^\top \hat{\mathbf{R}} \mathbf{v}_i \right)
]

这是最直接的物理残差。

### 17.7.2 像面残差

将解出的姿态用于重新投影参考星，看其与观测星质心的像面差：

[
\Delta x_i,\ \Delta y_i
]

角残差适合姿态一致性判断，像面残差适合定位误差归因与调参。

## 17.8 推荐的异常点剔除流程

第一阶段不建议一上来用很复杂的 RANSAC 变种。
更稳妥的实现是“三步法”：

### 第 1 步：初筛

在进入 QUEST 前，先剔除这些明显不该参与解算的匹配星：

- `low_snr`
- `edge_truncated`
- `saturated`
- `elongated`
- `match_score` 太低

### 第 2 步：首次解算

用全部剩余星做一次加权 QUEST。

### 第 3 步：解后残差剔除并重解

按残差做 sigma clipping 或百分位裁剪：

- 若 ( r_i > k \cdot \mathrm{RMS} )，剔除；
- 或若 ( r_i > r_{\max} )，剔除；
- 重解一次；
- 若二次重解仍失败，则输出降级/无效。

这样既不复杂，也足够工程化。

## 17.9 四元数、方向余弦矩阵、欧拉角的使用原则

这一点必须在文档里写死，不然后续代码会乱。

### 原则 1：内部主变量只用四元数

DSP 任务书已经明确要求：姿态基本表示应为四元数，不得以欧拉角作为内部主变量。

### 原则 2：方向余弦矩阵作为缓存量

矩阵适合做投影、残差和坐标变换，但不应作为主状态长期传播。

### 原则 3：欧拉角只用于输出和可视化

欧拉角适合：

- 调试打印
- 误差曲线展示
- 与外部系统接口做对比

但绝不能作为内部累计更新变量。

## 17.10 数值稳定性约束

DSP 任务书专门强调了 QUEST 的数值稳定性与异常判据：几何分布退化时不得产生非物理解，四元数结果必须归一化；若 QUEST 失败或特征值不满足判据，应将该帧结果标记为无效或降级。

因此 `fsglib` 中至少要做以下检查：

1. 四元数模长接近 1
2. 解矩阵正交性误差足够小
3. 残差 RMS 在阈值内
4. 匹配星分布不退化
5. 相对上一帧姿态变化不过于非物理
6. 若在跟踪模式下，当前解与预测解差异超限，则标记怀疑

## 17.11 1–3 片探测器失效情况下的降级解算

你已经明确要求从第一版开始就支持“4 片联合导星，同时支持 1–3 片失效”。

这里建议非常明确地采用“统一算法 + 子集降级”的思想：

- 不存在单独的一套“单片模式算法”；
- 只是在构造 `matched_stars` 时，可用星来自部分探测器子集；
- 质量评估中单独记录：
  - `num_detectors_active`
  - `stars_per_detector`
  - `geometry_spread_score`

推荐定义降级等级：

- `NORMAL_4D`：4 片均有效
- `DEGRADED_3D`：3 片有效
- `DEGRADED_2D`：2 片有效
- `DEGRADED_1D`：1 片有效
- `LOST`：不足以稳定解算

其中 `DEGRADED_1D` 不等于一定无效，因为如果单片星数足够多、几何分布足够好，仍可能给出可用解，但必须提高有效性门槛。

## 17.12 推荐接口

```python
def solve_attitude(
    solve_input: AttitudeSolveInput,
    cfg: dict
) -> AttitudeSolution:
    ...
def compute_weights(
    matched_stars: list[MatchedStar],
    cfg: dict
) -> np.ndarray:
    ...
def compute_residuals(
    q_ib: np.ndarray,
    matched_stars: list[MatchedStar],
    cfg: dict
) -> dict:
    ...
def reject_outliers(
    matched_stars: list[MatchedStar],
    residuals: dict,
    cfg: dict
) -> list[MatchedStar]:
    ...
```

## 17.13 解算结果输出规范

输出字段建议固定包括：

- `q_ib`
- `c_ib`
- `euler_zyx`
- `valid`
- `mode`
- `num_matched`
- `num_rejected`
- `residual_rms_arcsec`
- `residual_max_arcsec`
- `quality_flag`
- `degraded_level`
- `active_detector_ids`
- `solver_iterations`

其中：

- `q_ib` 是唯一强制主输出；
- `euler_zyx` 只是调试输出；
- `quality_flag` 要能区分成功、降级、无效、失锁。

DSP 任务书也明确要求输出最近一次姿态解算状态、有效匹配星数、平均残差以及异常计数等状态信息。

## 17.14 连续失败、重捕获与恢复

这一部分不能只写在“系统状态机”里，也要在算法文档里写清楚。

DSP 任务书给出的逻辑是：

- 当匹配星数低于阈值或残差显著增大时，触发重捕获或退回初始姿态建立模式；
- 在星跟踪模式下连续解算失败达到设定次数时，应自动切换到初始姿态建立模式；
- 当初始姿态建立也连续失败时，应进入安全状态或待机，并在恢复时重新初始化相关算法状态。

因此 `fsglib` 里建议定义统一状态机计数器：

```python
@dataclass
class SolveStateMachine:
    mode: str
    consecutive_match_failures: int
    consecutive_solve_failures: int
    consecutive_init_failures: int
    last_valid_q_ib: np.ndarray | None
    last_valid_time_s: float | None
```

---

# 18. 误差传播与统一验证框架

## 18.1 为什么必须单独建立这一层

你已经明确说了，这次最关心的主指标是姿态解算精度，但同时必须输出各层指标。这个判断完全正确。

因为现有性能验证方案本身就不是只看最终姿态，而是先把单星定位精度分解成：

- TE
- HSFE
- LSFE

然后再组合为单星定位精度。验证方案明确给出：

[
\sigma_s^2 = TE^2 + HSFE^2 + LSFE^2
]

并要求外场测试同时关注星图识别算法有效性、识别恒星数目与算法实时性。

所以 `fsglib` 不能只算“最后姿态误差多少”，还必须搭一个统一的误差归因框架。

## 18.2 TE、HSFE、LSFE 在 `fsglib` 中的对应实现

### 18.2.1 TE：瞬时误差

验证方案中，TE 被定义为随机误差，来源包括暗电流、读出噪声、ADC 量化噪声、电路开关噪声和杂散光；地面验证中通过在多个位置重复采样同一星点质心散布来估计。

在 `fsglib` 中，TE 的代理量建议定义为：

- 同一仿真条件下重复 Monte Carlo 注噪后的质心标准差；
- 同一星在同一亚像元相位下重复采样的质心 RMS；
- 单帧质心误差与 SNR 的经验关系。

### 18.2.2 HSFE：高频空间误差

验证方案中，HSFE 与热变化造成的焦平面偏移、探测器像元物理特性导致的动态残差、以及暗信号/响应非一致性有关，并通过微步距扫描不同成像位置来估计。

在 `fsglib` 中，HSFE 建议通过“亚像素栅格扫描”来建模：

- 固定光学 PSF
- 在一个像素内做细网格位移
- 用相同质心算法反演位置
- 统计偏差的高频结构

这其实就是你的 Python 验证版极应该做的一个关键专项。

### 18.2.3 LSFE：低频误差

验证方案中，LSFE 是通过跨视场位置变化、对误差面做拟合后，再估算低频残差得到。

在 `fsglib` 中，LSFE 建议定义为：

- 视场位置相关的系统性定位偏差；
- 畸变模型残差；
- 大尺度焦面热漂移带来的系统误差；
- 探测器拼接与安装误差未完全补偿后的慢变误差。

实现上可以通过：

1. 在全视场均匀采样大量真值星点；
2. 求出测得质心与真值位置偏差场；
3. 拟合低阶多项式或样条面；
4. 用拟合前后残差分离出低频/高频部分。

## 18.3 单星定位误差到姿态误差的传播

验证方案给出的系统级思路非常明确：单像元角分辨率约 3″，单星定位需达到不低于 1/5 像素，再通过 4 片探测器总计约 400 颗恒星支撑姿态精度达到 ≤30 mas（1σ）。

同时，已有仿真结论表明，在 400 颗导星条件下，平动相对精度约可达到 0.0187″，旋向相对精度约 0.0473″，能够满足凌星导星指标要求。

因此在当前阶段，姿态层验证报告应至少同时对照以下两套门槛：

- 凌星相机单帧门槛：非绕光轴 `<= 0.02"`，绕光轴 `<= 0.05"`
- 凌星相机多帧叠加门槛：1 min 内非绕光轴 `<= 0.003"`，绕光轴 `<= 0.01"`

因此在 `fsglib` 中，误差传播层至少要同时输出两级结果：

### 级别 A：单星层

- `sigma_te_pixel`
- `sigma_hsfe_pixel`
- `sigma_lsfe_pixel`
- `sigma_single_star_pixel`
- `sigma_single_star_arcsec`

### 级别 B：姿态层

- `sigma_pitch_arcsec`
- `sigma_yaw_arcsec`
- `sigma_roll_arcsec`
- `sigma_transverse_arcsec`
- `sigma_rotational_arcsec`

## 18.4 建议的误差传播实现方式

### 第一阶段：经验蒙特卡洛传播

最适合你当前阶段。

做法：

1. 从真值姿态和真值星表生成仿真星图；
2. 跑完整 `fsglib` 链路；
3. 引入不同误差源开关：
   - 只加 TE
   - 只加 HSFE
   - 只加 LSFE
   - 全部叠加
4. 统计最终三轴姿态误差分布。

### 第二阶段：局部线性传播

在姿态解附近，对观测向量误差做一阶线性化，得到近似协方差传播。
这一步更适合写成分析工具，不必强行嵌到主流程里。

## 18.5 必须输出的误差报告

建议 `fsglib/metrics/` 最终能统一产出以下报告对象：

```python
@dataclass
class ErrorBudgetReport:
    te_pixel: float
    hsfe_pixel: float
    lsfe_pixel: float
    single_star_pixel: float
    single_star_arcsec: float
    attitude_sigma_pitch_arcsec: float
    attitude_sigma_yaw_arcsec: float
    attitude_sigma_roll_arcsec: float
    num_stars_used_mean: float
    identification_success_rate: float
    runtime_ms_mean: float
    runtime_ms_p95: float
```

---

# 19. `fsglib` 端到端流程设计

## 19.1 单帧初始识别流程

```text
raw npz
 -> preprocess
 -> extract stars
 -> build reference stars (init mode)
 -> initial matching
 -> solve attitude
 -> validate solution
 -> save metrics/debug artifacts
```

## 19.2 多帧星跟踪流程

```text
frame_t0:
  init acquisition -> attitude q0

frame_t1...tn:
  preprocess
  -> extract
  -> ephemeris prediction from q_(t-1)
  -> tracking association
  -> solve attitude
  -> residual/outlier rejection
  -> state machine update
```

## 19.3 推荐顶层接口

```python
def run_single_frame_init(npz_path: str, cfg: dict) -> dict:
    ...
def run_sequence_tracking(npz_paths: list[str], cfg: dict) -> dict:
    ...
def evaluate_dataset(dataset_root: str, cfg: dict) -> dict:
    ...
```

---

# 20. 测试计划与验收建议

## 20.1 单元测试

每个模块必须有独立单元测试：

- `test_preprocess.py`
- `test_extract.py`
- `test_match.py`
- `test_ephemeris.py`
- `test_attitude.py`

要验证的不是“程序能不能跑”，而是：

- 数学口径是否一致；
- 输入输出结构是否稳定；
- 边界条件是否可控；
- 调参是否不会破坏其他模块。

## 20.2 集成测试

最少要有这 5 类集成测试：

1. 单探测器单帧
2. 四探测器单帧联合
3. 初始识别成功链路
4. 连续多帧跟踪链路
5. 故障降级链路

## 20.3 蒙特卡洛测试

这个对你们非常重要。因为你们的目标不是“看一个样例能跑通”，而是“验证这条方法在统计意义上能否稳定达到精度”。

建议参数扫描维度包括：

- 星等
- SNR
- 离焦量
- PSF 位置
- 背景水平
- 坏点比例
- 匹配错配率
- 有效探测器数
- 视场位置
- 先验姿态误差

已有仿真报告也表明，离焦、星等、可用星数都会显著影响定位精度与最终姿态精度。

## 20.4 实时性测试

性能验证方案明确把算法实时性列为外场验证目标之一；DSP 任务书则进一步给出了板上要求：更新频率不低于 4 Hz，单帧处理延迟不超过 200 ms。

对当前 PC 版 `fsglib`，我建议不要把“必须小于 200 ms”作为硬验收线，但必须做两件事：

1. 输出各模块耗时；
2. 输出全链路平均耗时和 P95 耗时。

因为这会直接告诉你：

- 以后板上迁移时哪个模块最危险；
- 初始识别和跟踪模式谁更重；
- 是否需要提前对星表查询、匹配、投影做结构优化。

## 20.5 推荐验收指标

针对第一阶段算法验证版，建议设立以下验收口径：

### A. 功能性

- 5 个模块均可独立运行
- 初始识别链路可闭环
- 跟踪链路可连续运行
- 支持 4 片联合与部分失效降级

### B. 正确性

- 与仿真真值可对比
- 匹配关系可解释
- 姿态输出稳定
- 无非物理四元数输出

### C. 可评估性

- TE/HSFE/LSFE 代理量可计算
- 单星定位误差可统计
- 姿态误差三轴分量可统计
- 成功率、星数、耗时可统计

### D. 可扩展性

- 星表可替换
- 光学模型可替换
- 权重函数可替换
- 匹配策略可替换

---

# 21. 面向 Copilot 的模块级开发提示模板

下面这部分你后面可以直接拆出来，作为仓库中的 Copilot 参考提示。

## 21.1 模块 4：星历查找

> 为 `fsglib.ephemeris` 编写纯 Python + NumPy/SciPy 的实现。
> 目标：根据时刻、先验姿态、4 片探测器几何与光学模型，从 Gaia 星表中筛选当前可见参考星，并预测其在各探测器上的像面位置。
> 必须支持 `init` 和 `tracking` 两种模式。
> 必须把时间推进、观测修正、像面投影拆成独立函数。
> 禁止在核心逻辑中做绘图。
> 需要类型注解、dataclass、单元测试。
> 输出应包含参考星方向矢量、预测像面坐标、是否越界、亮度与权重提示。

## 21.2 模块 5：姿态解算

> 为 `fsglib.attitude` 编写纯 Python + NumPy/SciPy 的实现。
> 目标：根据已匹配的观测星—参考星方向矢量，使用加权 QUEST 解算姿态四元数。
> 必须以四元数作为内部主变量，欧拉角仅作为输出。
> 必须支持权重计算、残差计算、异常点剔除、重解、质量评估。
> 必须支持 4 片联合与部分探测器失效场景。
> 需要输出残差 RMS、最大残差、剔除星数、有效标志、降级标志。
> 禁止输出未经校验的非单位四元数。

## 21.3 端到端管线

> 为 `fsglib.examples` 编写端到端示例：
>
> 1. 单帧初始识别
> 2. 多帧星跟踪
>    输入为 `.npz` 仿真星图。
>    输出为：中间结果、最终姿态、各模块耗时、识别成功率、匹配星数、残差统计。
>    所有阈值来自配置文件，不得硬编码。

---

很好，下面继续把第三部分写成真正可以开工的版本。这里的目标不是再讲概念，而是把 `fsglib` 直接推进到“今天就能建仓、明天就能写代码、这周就能跑第一条链路”的程度。

---

# 23. 开发目标重述：首版到底要先做成什么样

`fsglib` 首版不追求一步到位。最合理的目标是：

第一，先把“单帧初始识别链路”跑通。
也就是：读入一帧仿真星图，完成预处理、星点提取、参考星生成、初始匹配、姿态解算，并输出中间结果和误差指标。

第二，再把“多帧跟踪链路”跑通。
也就是：把上一帧姿态作为当前帧先验，完成导航星预测、局部匹配和连续姿态更新。

第三，在此基础上建立可量化的评估与回归机制。
这一步比“让程序跑起来”更重要，因为你们最终不是交一个 demo，而是要证明算法有效性和解算精度闭环成立。

所以，首版最小验收标准建议定成这样：

1. 能读 `.npz` 仿真星图；
2. 能完成单片和四片的统一接口处理；
3. 能输出星点候选列表；
4. 能输出一组参考星；
5. 能完成至少一种初始识别；
6. 能用 QUEST 输出姿态四元数；
7. 能输出残差、匹配星数、耗时；
8. 能保存调试产物。

---

# 24. 仓库初始化建议

## 24.1 顶层目录

第一天建仓时，建议直接按下面这个目录落地，不要再抽象讨论。

```text
fsglib/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ base.yaml
│  ├─ transit.yaml
│  ├─ microlensing.yaml
│  └─ detector_layout.yaml
├─ data/
│  ├─ samples/
│  ├─ truth/
│  └─ cache/
├─ docs/
│  ├─ architecture.md
│  ├─ development_guide.md
│  └─ copilot_notes.md
├─ fsglib/
│  ├─ __init__.py
│  ├─ common/
│  ├─ preprocess/
│  ├─ extract/
│  ├─ match/
│  ├─ ephemeris/
│  ├─ attitude/
│  └─ pipeline/
├─ tests/
└─ examples/
```

这里有一个原则：
`docs/` 是给人看的，`configs/` 是给程序调参的，`examples/` 是给第一次跑链路用的，`tests/` 是给你们以后不返工用的。

## 24.2 `requirements.txt`

你已经明确要求只用 `Python + NumPy + SciPy`，所以首版依赖尽量克制：

```text
numpy
scipy
pyyaml
pytest
matplotlib
```

这里我加 `pyyaml` 和 `pytest` 是必要的，`matplotlib` 只是为了调试可视化，不进入核心依赖逻辑。

不要在第一版引入：

- opencv
- pandas
- astropy
- scikit-image

不是说它们不能用，而是现在你们最重要的是把算法主链和数据结构钉死，依赖越少，越容易控制。

## 24.3 `pyproject.toml`

建议最简单版本即可：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "fsglib"
version = "0.1.0"
description = "Fine Star Guiding Library for ET algorithm verification"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

---

# 25. 首版配置文件该怎么写

这一部分很关键。
`fsglib` 能不能后面稳定扩展，很大程度取决于你一开始有没有把配置结构设计清楚。

## 25.1 `configs/base.yaml` 建议字段

下面给出一个建议版骨架。第一天就可以直接建这个文件。

```yaml
project:
  name: fsglib
  mode: init
  save_debug: true
  output_dir: outputs/debug

io:
  npz_image_key: images
  npz_time_key: time_s
  npz_variant_key: variant_ids
  npz_cadence_key: cadence_s
  npz_unit_key: unit

detector:
  num_detectors: 4
  image_height: 1119
  image_width: 1119
  pixel_size_um: null
  saturation_value: null
  bad_pixel_map: null

preprocess:
  enable_bias_subtraction: false
  enable_dark_subtraction: false
  enable_flat_field: false
  enable_bad_pixel_mask: false
  enable_background_subtraction: true
  background_method: sigma_clip_global
  sigma_clip_k: 3.0
  denoise_method: none

extract:
  detection_image: snr
  seed_threshold_sigma: 5.0
  grow_threshold_sigma: 3.0
  min_area: 3
  max_area: 200
  centroid_method: weighted_centroid
  bbox_expand: 2
  reject_edge_margin: 3
  max_ellipticity: 0.8

match:
  mode: init
  init_max_catalog_radius_deg: 8.0
  init_bright_star_topk: 20
  pair_angle_tol_arcsec: 120.0
  hypothesis_topk: 50
  validate_max_residual_arcsec: 60.0
  validate_min_support: 5

tracking:
  search_radius_pix: 10.0
  max_miss_count: 3
  reacquire_after_failures: 2

ephemeris:
  catalog_backend: mock
  mag_limit: 15.0
  enable_proper_motion: true
  enable_precession: false
  enable_nutation: false
  enable_dva: false
  enable_relativity: false

attitude:
  solver: quest
  min_stars_mathematical: 2
  min_stars_operational: 4
  weight_mode: variance_snr_hybrid
  outlier_reject_enable: true
  outlier_max_residual_arcsec: 30.0
  outlier_sigma_clip: 3.0
  max_iterations: 2

metrics:
  enable_truth_compare: true
  report_centroid_error: true
  report_attitude_error: true
  report_runtime: true

logging:
  level: INFO
  save_intermediate_arrays: true
  save_source_catalog: true
  save_match_result: true
```

这个配置结构的优点是：
层次干净，而且和 5 个模块是一一对应的。

## 25.2 `transit.yaml` 和 `microlensing.yaml`

这两个文件不应该复制整份配置，只覆盖不同参数。例如：

```yaml
project:
  mode: tracking

extract:
  seed_threshold_sigma: 4.5
  grow_threshold_sigma: 2.8

tracking:
  search_radius_pix: 8.0
```

以及：

```yaml
project:
  mode: init

extract:
  seed_threshold_sigma: 5.5
  grow_threshold_sigma: 3.5

match:
  init_bright_star_topk: 30
```

这样以后你们切模式只是在加载配置时叠加 override，而不是维护两套配置宇宙。

## 25.3 `detector_layout.yaml`

这个文件应该专门描述 4 片探测器的布局，不要混在 `base.yaml` 里。建议至少包括：

```yaml
layout:
  frame_name: fgs_body
  detectors:
    - detector_id: 0
      name: det0
      active: true
      principal_point_pix: [559.5, 559.5]
      mounting_matrix: null
      fov_boundary_lb: null
    - detector_id: 1
      name: det1
      active: true
      principal_point_pix: [559.5, 559.5]
      mounting_matrix: null
      fov_boundary_lb: null
    - detector_id: 2
      name: det2
      active: true
      principal_point_pix: [559.5, 559.5]
      mounting_matrix: null
      fov_boundary_lb: null
    - detector_id: 3
      name: det3
      active: true
      principal_point_pix: [559.5, 559.5]
      mounting_matrix: null
      fov_boundary_lb: null
```

这里先允许 `null`，因为你还在向光学工程师收数据。
但是字段名字必须先定下来。

---

# 26. `.npz` 数据读入器怎么写

## 26.1 设计原则

你现在已知仿真输入结构是：

- `images`
- `variant_ids`
- `coadd_start`
- `coadd_stop`
- `time_s`
- `cadence_s`
- `unit`

所以第一版一定不要把读入器写成“一次性临时脚本”。
应该直接封成 `fsglib/common/io.py` 中的正式接口。

## 26.2 推荐实现接口

```python
def load_npz_frame(npz_path: str, detector_id: int = 0) -> RawFrame:
    ...
```

以及批量版本：

```python
def load_npz_sequence(npz_paths: list[str], detector_id: int = 0) -> list[RawFrame]:
    ...
```

## 26.3 最低功能要求

读入器必须做这些事：

1. 检查文件是否存在；
2. 检查必要 key 是否存在；
3. 把 `(1, 1, H, W)` 归一成 `(H, W)`；
4. 把标量从 numpy scalar 转成 python 原生类型；
5. 记录原始 meta；
6. 报错信息要明确，不要只抛一个 `IndexError` 或 `KeyError`。

## 26.4 推荐代码骨架

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from fsglib.common.types import RawFrame

def load_npz_frame(npz_path: str, detector_id: int = 0) -> RawFrame:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(path, allow_pickle=False)
    required = ["images", "time_s"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in {npz_path}")

    image = data["images"]
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    time_s = float(np.atleast_1d(data["time_s"])[0])

    variant_id = None
    if "variant_ids" in data:
        variant_id = int(np.atleast_1d(data["variant_ids"])[0])

    cadence_s = None
    if "cadence_s" in data:
        cadence_s = float(data["cadence_s"])

    coadd_start = int(data["coadd_start"]) if "coadd_start" in data else None
    coadd_stop = int(data["coadd_stop"]) if "coadd_stop" in data else None
    unit = str(data["unit"]) if "unit" in data else None

    return RawFrame(
        detector_id=detector_id,
        image=np.asarray(image, dtype=np.float64),
        time_s=time_s,
        cadence_s=cadence_s,
        coadd_start=coadd_start,
        coadd_stop=coadd_stop,
        unit=unit,
        variant_id=variant_id,
        meta={"npz_path": str(path)}
    )
```

这里我建议在核心链路中统一转成 `float64`。
不是因为必须，而是第一阶段你们更关心数值稳定和调试方便，而不是极限性能。

---

# 27. 第一批必须先写的 dataclass

在第一周内，建议优先把以下 dataclass 全部写完。
这是整个库的骨架，比算法细节还重要。

## 27.1 `fsglib/common/types.py`

你们至少需要这几类：

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RawFrame:
    detector_id: int
    image: np.ndarray
    time_s: float
    cadence_s: float | None = None
    coadd_start: int | None = None
    coadd_stop: int | None = None
    unit: str | None = None
    variant_id: int | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class PreprocessedFrame:
    detector_id: int
    image: np.ndarray
    background: np.ndarray | float
    noise_map: np.ndarray | float
    valid_mask: np.ndarray
    preprocess_meta: dict = field(default_factory=dict)

@dataclass
class StarCandidate:
    detector_id: int
    source_id: int
    x: float
    y: float
    flux: float
    peak: float
    area: int
    snr: float
    bbox: tuple[int, int, int, int]
    shape: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

@dataclass
class ObservedStar:
    detector_id: int
    source_id: int
    x: float
    y: float
    los_body: np.ndarray
    flux: float
    snr: float
    weight: float = 1.0
    flags: dict = field(default_factory=dict)

@dataclass
class MatchedStar:
    detector_id: int
    source_id: int
    catalog_id: int
    los_body: np.ndarray
    los_inertial: np.ndarray
    residual_arcsec: float | None = None
    weight: float = 1.0
    match_score: float = 0.0
    flags: dict = field(default_factory=dict)

@dataclass
class AttitudeSolution:
    q_ib: np.ndarray
    c_ib: np.ndarray | None
    euler_zyx: np.ndarray | None
    valid: bool
    mode: str
    num_matched: int
    residual_rms_arcsec: float
    residual_max_arcsec: float
    quality: dict = field(default_factory=dict)
```

## 27.2 为什么 dataclass 要先写完

因为后面无论 Copilot 还是同事写代码，只要输入输出对象不乱，整个工程就不会乱。

真正最怕的不是算法慢一点，而是每个模块传参风格都不同：

- 有的传 `dict`
- 有的传 `tuple`
- 有的传 `numpy array`
- 有的临时拼 `list`

这种东西一旦开始，就会很快失控。

---

# 28. 五个模块的首版代码骨架

下面这部分最适合直接作为“开工模板”。

# 28.1 模块 1：预处理

文件：`fsglib/preprocess/pipeline.py`

```python
import numpy as np
from fsglib.common.types import RawFrame, PreprocessedFrame

def preprocess_frame(raw: RawFrame, calib: dict, cfg: dict) -> PreprocessedFrame:
    image = raw.image.copy()
    valid_mask = np.isfinite(image)

    image = np.where(valid_mask, image, 0.0)

    if cfg["preprocess"].get("enable_background_subtraction", True):
        background = estimate_background(image, valid_mask, cfg)
        image_sub = image - background
    else:
        background = 0.0
        image_sub = image

    noise_map = estimate_noise_map(image_sub, valid_mask, cfg)

    return PreprocessedFrame(
        detector_id=raw.detector_id,
        image=image_sub,
        background=background,
        noise_map=noise_map,
        valid_mask=valid_mask,
        preprocess_meta={}
    )

def estimate_background(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    median = np.median(vals)
    return median

def estimate_noise_map(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    sigma = np.std(vals)
    return np.full_like(image, fill_value=max(sigma, 1e-6), dtype=np.float64)
```

这个版本很简单，但已经足够做第一条链路。

# 28.2 模块 2：星点提取

文件：`fsglib/extract/pipeline.py`

```python
import numpy as np
from scipy import ndimage
from fsglib.common.types import PreprocessedFrame, StarCandidate

def extract_stars(frame: PreprocessedFrame, cfg: dict) -> list[StarCandidate]:
    image = frame.image
    noise = frame.noise_map
    mask = frame.valid_mask

    snr_map = np.where(mask, image / np.maximum(noise, 1e-6), 0.0)

    seed_th = cfg["extract"]["seed_threshold_sigma"]
    detect_mask = snr_map > seed_th

    labeled, num = ndimage.label(detect_mask)
    candidates = []

    for label_id in range(1, num + 1):
        seg = labeled == label_id
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue

        area = len(xs)
        if area < cfg["extract"]["min_area"] or area > cfg["extract"]["max_area"]:
            continue

        flux = float(np.sum(image[seg]))
        peak = float(np.max(image[seg]))

        if flux <= 0:
            continue

        x = float(np.sum(xs * image[seg]) / flux)
        y = float(np.sum(ys * image[seg]) / flux)

        snr = float(np.sum(image[seg]) / np.sqrt(np.sum(noise[seg] ** 2)))

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        candidates.append(
            StarCandidate(
                detector_id=frame.detector_id,
                source_id=len(candidates),
                x=x,
                y=y,
                flux=flux,
                peak=peak,
                area=area,
                snr=snr,
                bbox=bbox,
                shape={},
                flags={}
            )
        )

    return candidates
```

# 28.3 模块 3：星图识别

文件：`fsglib/match/pipeline.py`

```python
from fsglib.common.types import ObservedStar, MatchedStar

def match_stars_init(
    observed_stars: list[ObservedStar],
    reference_stars: list,
    cfg: dict
) -> list[MatchedStar]:
    matched = []

    for obs in observed_stars:
        best = None
        best_dist = None

        for ref in reference_stars:
            if obs.detector_id not in ref.predicted_xy:
                continue
            pred_x, pred_y = ref.predicted_xy[obs.detector_id]
            dx = obs.x - pred_x
            dy = obs.y - pred_y
            dist2 = dx * dx + dy * dy

            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best = ref

        if best is None:
            continue

        matched.append(
            MatchedStar(
                detector_id=obs.detector_id,
                source_id=obs.source_id,
                catalog_id=best.catalog_id,
                los_body=obs.los_body,
                los_inertial=best.los_inertial,
                residual_arcsec=None,
                weight=1.0,
                match_score=1.0,
                flags={}
            )
        )

    return matched
```

这个版本当然很粗，但足以构成第一条“先验投影 + 最近邻匹配”的初始原型。

# 28.4 模块 4：星历查找

文件：`fsglib/ephemeris/pipeline.py`

```python
import numpy as np

def build_reference_stars(ctx, catalog_provider, projector, cfg):
    if ctx.mode == "init":
        catalog_stars = catalog_provider.query_region(
            boresight=ctx.prior_attitude_q,
            radius_deg=cfg["match"]["init_max_catalog_radius_deg"],
            mag_limit=cfg["ephemeris"]["mag_limit"]
        )
    else:
        catalog_stars = catalog_provider.query_tracking_targets(ctx)

    ref_stars = []
    for star in catalog_stars:
        los_inertial = radec_to_unit_vector(star.ra_deg, star.dec_deg)
        predicted_xy = projector.project_to_detectors(
            los_inertial=los_inertial,
            attitude_q=ctx.prior_attitude_q
        )
        ref_stars.append(
            projector.make_reference_star(
                star=star,
                time_s=ctx.time_s,
                los_inertial=los_inertial,
                predicted_xy=predicted_xy
            )
        )
    return ref_stars

def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float64)
```

# 28.5 模块 5：姿态解算

文件：`fsglib/attitude/solver.py`

```python
import numpy as np
from fsglib.common.types import AttitudeSolution

def solve_attitude(matched_stars: list, cfg: dict) -> AttitudeSolution:
    if len(matched_stars) < cfg["attitude"]["min_stars_mathematical"]:
        return AttitudeSolution(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            c_ib=None,
            euler_zyx=None,
            valid=False,
            mode="lost",
            num_matched=len(matched_stars),
            residual_rms_arcsec=np.inf,
            residual_max_arcsec=np.inf,
            quality={"reason": "not_enough_stars"}
        )

    q_ib = solve_quest(matched_stars, cfg)
    c_ib = quat_to_dcm(q_ib)

    residuals = compute_residuals(c_ib, matched_stars)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    rmax = float(np.max(residuals))

    valid = len(matched_stars) >= cfg["attitude"]["min_stars_operational"]

    return AttitudeSolution(
        q_ib=q_ib,
        c_ib=c_ib,
        euler_zyx=dcm_to_euler_zyx(c_ib),
        valid=valid,
        mode="init",
        num_matched=len(matched_stars),
        residual_rms_arcsec=rms,
        residual_max_arcsec=rmax,
        quality={}
    )

def solve_quest(matched_stars: list, cfg: dict) -> np.ndarray:
    # 首版先允许用 Wahba/SVD 占位，后续再替换为严格 QUEST
    B = np.zeros((3, 3), dtype=np.float64)
    for m in matched_stars:
        w = np.asarray(m.los_body, dtype=np.float64)
        v = np.asarray(m.los_inertial, dtype=np.float64)
        B += np.outer(w, v)

    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return dcm_to_quat(R)
```

这里有一个很实际的建议：
首版可以先用 SVD/Wahba 占位，把整条链路打通。等全链路稳定后，再把 `solve_quest()` 换成严格 QUEST 实现。
不要一上来就在 QUEST 数值细节上消耗过多时间，导致全链路迟迟不闭环。

---

# 29. 顶层 pipeline 怎么组织

## 29.1 单帧初始链路

文件：`fsglib/pipeline/run_init.py`

```python
from fsglib.common.io import load_npz_frame
from fsglib.preprocess.pipeline import preprocess_frame
from fsglib.extract.pipeline import extract_stars
from fsglib.pipeline.convert import candidates_to_observed
from fsglib.ephemeris.pipeline import build_reference_stars
from fsglib.match.pipeline import match_stars_init
from fsglib.attitude.solver import solve_attitude

def run_single_frame_init(npz_path: str, cfg: dict, models: dict) -> dict:
    raw = load_npz_frame(npz_path, detector_id=0)
    pre = preprocess_frame(raw, calib=models.get("calib", {}), cfg=cfg)
    cand = extract_stars(pre, cfg=cfg)
    obs = candidates_to_observed(cand, models["projector"], cfg)

    eph_ctx = models["ephemeris_context_builder"](raw, cfg, models)
    ref = build_reference_stars(eph_ctx, models["catalog_provider"], models["projector"], cfg)

    matched = match_stars_init(obs, ref, cfg)
    solution = solve_attitude(matched, cfg)

    return {
        "raw": raw,
        "preprocessed": pre,
        "candidates": cand,
        "observed": obs,
        "reference": ref,
        "matched": matched,
        "solution": solution,
    }
```

## 29.2 为什么先只写 init pipeline

因为最先要解决的是“能不能闭环”。
而跟踪模式本质上是在 init 闭环成功之后，再加上状态延续和预测逻辑。

所以开发顺序必须是：

1. 单帧 init 跑通；
2. init 结果可解释；
3. 再写 tracking。

---

# 30. 第一周的开发顺序建议

这里我给你一个非常具体的顺序。
照这个顺序推进，效率会比“按模块各写一点”高得多。

## 第 1 天：建仓和数据骨架

完成这些事：

- 建仓库目录；
- 写 `requirements.txt`；
- 写 `base.yaml`；
- 写 `RawFrame / PreprocessedFrame / StarCandidate / MatchedStar / AttitudeSolution`；
- 写 `.npz` 读入器；
- 用一帧样例把数据读出来并打印关键字段。

当天的验收标准：
`python examples/check_npz.py` 能正确打印图像 shape、time、variant_id、unit。

## 第 2 天：预处理和提取跑通

完成这些事：

- 写 `preprocess_frame()` 最小实现；
- 写 `extract_stars()` 最小实现；
- 画出检测结果叠加图；
- 保存候选星 csv 或 json。

当天的验收标准：
对一帧仿真图像，能输出一批肉眼基本合理的星点候选。

## 第 3 天：像面到视线、参考星占位器

完成这些事：

- 写 `candidates_to_observed()`；
- 写 `radec_to_unit_vector()`；
- 写一个 `MockCatalogProvider`；
- 写一个最简 `projector` 占位器。

当天的验收标准：
能从人工构造的少量参考星生成预测位置，并和观测星做最近邻匹配。

## 第 4 天：初始匹配 + 姿态解算闭环

完成这些事：

- 写 `match_stars_init()`；
- 写 `solve_attitude()`；
- 先用 SVD/Wahba 占位；
- 输出四元数和残差。

当天的验收标准：
单帧链路从 `.npz` 到 `AttitudeSolution` 能完整返回结果。

## 第 5 天：中间结果保存与评估输出

完成这些事：

- 保存预处理图；
- 保存提取源列表；
- 保存匹配结果；
- 输出模块耗时；
- 输出残差统计；
- 输出成功/失败原因。

当天的验收标准：
一次运行后，`outputs/debug/` 里能看到完整调试产物。

## 第 6–7 天：整理回归样例并补测试

完成这些事：

- 固定 3–5 帧样例；
- 写最小单元测试；
- 写一份 `README.md` 里的运行说明；
- 记录已知问题。

当天的验收标准：
新成员拉下代码后，按 README 能复现同样的结果。

---

# 31. 第二周该做什么

第一周只追求闭环。
第二周才开始追求“像样”和“可信”。

建议第二周重点做这几件事：

第一，把 `MockCatalogProvider` 换成真实 Gaia 星表接口壳层。
哪怕一开始先读裁剪后的小表，也比一直用 mock 强。

第二，把 projector 从“简单针孔占位”升级到“支持每片探测器独立模型”。

第三，把初始匹配从“纯最近邻”升级到“先验子天区 + 候选验证”。

第四，把姿态求解从 SVD 占位升级到严格 QUEST。

第五，加上 outlier rejection 和重解。

第六，开始做多帧 tracking 原型。

---

# 32. 你们最容易踩的坑

这一段我直接给你列出来，都是非常典型、而且非常容易耽误进度的坑。

## 32.1 一开始就追求“完美光学模型”

这是第一大坑。
你们现在真正需要的是“能跑 + 可替换”。
所以 projector 第一版必须允许非常简化，只要接口正确即可。

## 32.2 一开始就下载全量 Gaia 并做复杂建库

这是第二大坑。
第一周根本不需要。
第一周只需要一个小的裁剪样本表，或者人工 mock 的参考星集合，把数据流跑通就够了。

## 32.3 一开始就把 QUEST 写到极致

这也是坑。
先用 SVD/Wahba 占位把整个链路跑起来，再替换成 QUEST，效率更高。

## 32.4 每个人各写各的，不统一 dataclass 和 config

这是最大的软件工程坑。
一旦发生，后面集成会非常痛苦。

## 32.5 在核心函数里直接画图

这会迅速把代码污染。
核心函数只返回结果，绘图统一放到 `examples/` 或 `debug/` 辅助模块。

---

# 33. 第一个可运行版本应该输出什么

首版跑完后，我建议在 `outputs/debug/<case_name>/` 下至少有这些东西：

```text
outputs/debug/case_001/
├─ raw.npy
├─ preprocessed.npy
├─ snr_map.npy
├─ candidates.json
├─ observed.json
├─ reference.json
├─ matched.json
├─ solution.json
├─ overlay_sources.png
├─ overlay_matches.png
└─ metrics.json
```

其中 `metrics.json` 至少要有：

```json
{
  "num_candidates": 123,
  "num_observed": 110,
  "num_reference": 185,
  "num_matched": 42,
  "attitude_valid": true,
  "residual_rms_arcsec": 12.3,
  "residual_max_arcsec": 38.7,
  "runtime_ms_total": 86.4
}
```

为什么这一步很重要？
因为后面你和同事在讨论问题时，就不会停留在“感觉提取得还行”“好像匹配错了几颗”，而是直接对着中间产物和指标说话。

---

# 34. 第一批单元测试该怎么写

首版测试不要追求覆盖率数字，追求“关键路径不炸”。

## 34.1 `tests/test_io.py`

至少测：

- 正常 `.npz` 能读；
- 缺 key 时能报清楚；
- `(1,1,H,W)` 能正确转成 `(H,W)`。

## 34.2 `tests/test_preprocess.py`

至少测：

- 背景估计返回合理标量或数组；
- `valid_mask` 正确屏蔽 NaN；
- 输出 shape 不变。

## 34.3 `tests/test_extract.py`

至少测：

- 构造一个人工高斯星点，能检测到；
- 质心位置误差在合理范围；
- 面积阈值能工作。

## 34.4 `tests/test_attitude.py`

至少测：

- 输入 2–3 对理想矢量时，能输出单位四元数；
- 构造一个已知旋转，解算结果接近真值；
- 匹配星数不足时，返回 `valid=False`。

---

# 35. 真实开工时我建议你们先写的 10 个文件

如果你想把任务直接分给同事，这 10 个文件最适合作为第一批开发任务：

1. `fsglib/common/types.py`
2. `fsglib/common/io.py`
3. `fsglib/preprocess/pipeline.py`
4. `fsglib/extract/pipeline.py`
5. `fsglib/common/coords.py`
6. `fsglib/pipeline/convert.py`
7. `fsglib/ephemeris/pipeline.py`
8. `fsglib/match/pipeline.py`
9. `fsglib/attitude/solver.py`
10. `examples/run_single_frame.py`

这 10 个文件一旦有了，首个闭环版本基本就出来了。

---

# 36. `examples/run_single_frame.py` 建议直接这样写

```python
from pathlib import Path
import yaml

from fsglib.pipeline.run_init import run_single_frame_init
from fsglib.common.debug import save_debug_bundle
from fsglib.models.mock import build_mock_models

def main():
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    npz_path = "data/samples/example_001.npz"

    models = build_mock_models(cfg)
    result = run_single_frame_init(npz_path=npz_path, cfg=cfg, models=models)

    save_debug_bundle(result, cfg)
    print("Attitude valid:", result["solution"].valid)
    print("Matched stars:", result["solution"].num_matched)
    print("Residual RMS (arcsec):", result["solution"].residual_rms_arcsec)

if __name__ == "__main__":
    main()
```

这就是你们第一天到第五天应该努力达到的效果。

---

# 37. 给 Copilot 的“首周开发提示词”

这个你后面可以直接丢给 Copilot，用来约束它。

## 37.1 类型与风格提示

> 请为 `fsglib` 项目编写纯 Python + NumPy/SciPy 代码。
> 所有核心输入输出都使用 dataclass，不使用未约束的 dict 作为主接口。
> 所有函数必须类型注解。
> 配置参数从 yaml 读取，不得硬编码到核心算法函数。
> 不要在核心函数内部做绘图。
> 所有模块应便于单元测试。
> 默认优先可读性与数值稳定性，不优先极限性能。

## 37.2 首版闭环提示

> 先实现最小可运行版本：
>
> 1. 读取 NPZ 仿真图像
> 2. 进行简单背景扣除
> 3. 基于阈值和连通域提取星点
> 4. 将候选星转换为观测视线向量
> 5. 使用 mock catalog 生成参考星
> 6. 基于最近邻完成初始匹配
> 7. 使用 Wahba/SVD 或 QUEST 输出姿态四元数
> 8. 输出匹配星数、残差 RMS、有效标记
>    先保证链路打通，再逐步替换占位实现。

## 37.3 单文件开发提示

比如你让 Copilot 写 `extract/pipeline.py` 时，可以用：

> 请为 `fsglib.extract.pipeline` 实现星点提取模块。
> 输入为 `PreprocessedFrame`，输出为 `list[StarCandidate]`。
> 首版采用 SNR 图阈值分割 + 连通域 + 加权质心法。
> 需要支持最小面积、最大面积、边缘剔除、SNR 计算。
> 返回的 `StarCandidate` 必须包含 x, y, flux, peak, area, snr, bbox, flags。
> 代码要清晰、可测试，不依赖 OpenCV 或 scikit-image。

---

# 38. 现在就可以分工的任务包

如果你们不是一个人单干，而是几个人并行推进，我建议这样拆：

A 同事负责数据与基础设施：
`types.py`、`io.py`、`config loader`、`debug saver`

B 同事负责图像侧：
`preprocess`、`extract`

C 同事负责几何与星表：
`coords.py`、`projector`、`ephemeris`

D 同事负责解算与评估：
`match`、`attitude`、`metrics`

你自己最适合盯两件事：

- dataclass 和 config 的统一；
- 各模块验收口径是否一致。

这两件事只要你抓牢，整个项目就不会散。

---

# 39. 第三部分的结论：现在该怎么开始

到这里，`fsglib` 已经不再只是一个抽象的算法库设想，而是已经具备明确开工路径了。

现在最合理的启动方式是：

先建仓，先定 dataclass，先定配置，先写 `.npz` 读入器。
然后优先把“单帧初始识别闭环”跑通。
不要一开始就纠缠真实 Gaia 全量接入、复杂畸变模型、严格 QUEST 数值优化这些次一级问题。
第一周的任务只有一个：让整条链路活起来，并且把中间结果看清楚。

等这一版闭环跑起来之后，你们再按顺序把占位实现替换掉：

- mock catalog 换真实 Gaia 子集
- 简化 projector 换真实 4 片模型
- 最近邻匹配换候选验证
- SVD 占位换严格 QUEST
- 单帧闭环换多帧 tracking
- 基础指标换完整误差预算

这样推进，节奏是对的，也最符合你现在“先证明算法设计有效性，再验证姿态解算精度”的目标。
