# `fsglib` API Document

## 1. 范围

用于ET导星算法验证。

__Author：chenxu__

__Version：v0.1.0__

## 2. 入口

1. `fsglib.pipeline.run_guide_first_frame_init`
2. `fsglib.pipeline.run_guide_first_frame_truth_noise`
3. ~~`fsglib.pipeline.run_init.run_single_frame_init`~~
4. ~~`fsglib.pipeline.run_tracking.run_sequence_tracking`~~
- `run_guide_first_frame_init` 和 `run_guide_first_frame_truth_noise` 导星四探测器联合验证链路的入口；
- ~~`run_single_frame_init` 和 `run_sequence_tracking` 通用的单帧序列处理接口，依赖外部 `models` 注入投影器和星表访问对象；~~

## 3. 当前链路分层

### 3.1 数据输入层

位置：

- [fsglib/common/io.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/io.py:1)

功能：

- 读取单帧 `npz`；
- 读取批次目录 `batch_root`；
- `DatasetContext`；
- 在可用时提取帧内 truth 信息；

### 3.2 图像到观测向量

位置：

- [fsglib/preprocess/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/preprocess/pipeline.py:1)
- [fsglib/extract/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/extract/pipeline.py:1)
- [fsglib/pipeline/convert.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/convert.py:1)
- [fsglib/pipeline/run_guide_init.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_init.py:1)
- [fsglib/pipeline/run_guide_truth_noise.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_truth_noise.py:1)

功能：

- 单帧，`RawFrame -> PreprocessedFrame -> StarCandidate -> ObservedStar`；
- 联合链路，`ObservedStar` 通过 `et_focalplane` ~~或拟合的 body model 构造~~（当前抛弃主点假设——chenxu）。

### 3.3 参考星、匹配与姿态解算层

位置：

- [fsglib/ephemeris/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/ephemeris/pipeline.py:1)
- [fsglib/match/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/match/pipeline.py:1)
- [fsglib/attitude/solver.py](/home/cxgao/ET/FSG/fsglib/fsglib/attitude/solver.py:1)

功能：

- 参考星构造；
- 预测像点或三角匹配；
- QUEST 解算、残差评估和异常点剔除。

### 3.4 评估与调试输出层

位置：

- [fsglib/pipeline/evaluate.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/evaluate.py:1)
- [fsglib/pipeline/guide_error_audit.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/guide_error_audit.py:1)
- [fsglib/common/debug.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/debug.py:772)

功能：

- 单帧评估；
- 序列汇总；
- 导星首帧误差拆分；
- 调试产物落盘。

## 4. 入口

### 4.1 `run_guide_first_frame_init(cfg) -> dict`

位置：

- [fsglib/pipeline/run_guide_init.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_init.py:471)

场景：

- 真实提取质心的导星首帧联合初始化；
- 4个 guide detector 联合；
- 参考星来自 `et_focalplane + GaiaCatalog`；
- 匹配和姿态解算 QUEST 实现。

输入：

- `cfg["guide_init"]` 完整；
- `cfg["et_coord"]` 提供 `src_dir`、`data_dir`、`gaia_root_dir`；
- `dataset_root` 下每个 batch 目录包含 `frames/*.npz`、`run_meta.json`，以及真值星表 `stars.ecsv`。

返回字典的Key：

- `solution`
- `matching`
- `observed_count`
- `reference_count`
- `detector_stats`
- `sim_to_detector_map`
- `geometry_model`
- `body_model`
- `error_audit`
- `meta`

说明：

- 先建立 sim 坐标到 `et_focalplane` detector 坐标的桥接，再做像点转 LOS；
- ~~`geometry_model` 和 `body_model` 当前返回的是同一套序列化结果~~。（不再使用body_model—chenxu）

### 4.2 `run_guide_first_frame_truth_noise(cfg) -> dict`

位置：

- [fsglib/pipeline/run_guide_truth_noise.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_guide_truth_noise.py:342)

适用场景：

- 不提取质心，直接从 truth detector 质心注入噪声；
- 用于评估质心误差、几何误差和姿态解算误差；
- `examples/run_guide_first_frame_truth_noise_exact.py` 。

### 4.3 `run_single_frame_init(npz_path, cfg, models, dataset_ctx=None) -> FrameResult`

位置：

- [fsglib/pipeline/run_init.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_init.py:112)

适用场景：

- 单探测器单帧初始化；

`models` ：

- `projector`
- `catalog`

可选：

- `calib`

返回值：

- `FrameResult`

说明：

- 默认使用 `candidates_to_observed`，依赖外部 `projector.pixel_to_los_body`；

### 4.4 `run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=None) -> SequenceResult`

位置：

- [fsglib/pipeline/run_tracking.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/run_tracking.py:237)

适用场景：

- 通用序列跟踪；
- 首帧走 init；
- 使用 `models` 提供投影和星表访问。

返回值：

- `SequenceResult`

产出：

- `mode_history`
- `state_history`
- `metrics`

## 5. 数据结构

 [fsglib/common/types.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/types.py:1)。

### 5.1 图像与候选星

`RawFrame`

- 单帧原始输入；

`PreprocessedFrame`

- 预处理后的图像、背景、噪声图和有效掩膜；
- 后续如果要引入暗电平/FPN/坏点图，在这里扩展——chenxu。

`StarCandidate`

- 星点提取的直接输出；
- `flags` 里写入质心窗口、peak 像素、bias correction ；

### 5.2 观测星与匹配星

`ObservedStar`

- 匹配和姿态解算的观测量；
- 字段是 `los_body`；
- `x/y` 保留 detector 像点，用于后续审计。

`MatchedStar`

- 观测星与参考星的配对；
- `los_body` 和 `los_inertial` 是姿态解算的输入；
- `flags["observed_xy"]`、`flags["predicted_xy"]`、`flags["residual_pix"]` （用于评估依赖——chenxu）。

### 5.3 上下文

`DatasetContext`

- 单帧数据上下文；
- 提供 `frame_paths`、静态 truth、field center、pixel scale、field offset 等；

`MatchingContext`

- 匹配器输入上下文；

`FrameResult`

- 单帧结果；
- 调试、评估。

`SequenceResult`

- 序列结果；
- `metrics` 的 `summarize_sequence_result` 。

### 5.4 星表相关

定义位置：

- [fsglib/ephemeris/types.py](/home/cxgao/ET/FSG/fsglib/fsglib/ephemeris/types.py:1)

对象：

- `CatalogStar`
- `ReferenceStar`
- `EphemerisContext`

`ReferenceStar` 的字段：

- `catalog_id`
- `los_inertial`
- `predicted_xy`
- `predicted_valid`
- `detector_ids_visible`

## 6. 模块接口

### 6.1 输入/装配

`load_npz_frame(npz_path, detector_id=0) -> RawFrame`

- [fsglib/common/io.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/io.py:377)
- 读取单帧；
- 从 `npz` 内取 truth payload；
- `images` 需为 `(1, 1, H, W)` 或 `(H, W)`。

`load_dataset_batch(batch_root, cfg=None) -> DatasetContext`

- [fsglib/common/io.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/io.py:252)
- 读取 `frames/`、`run_meta.json`、`stars.ecsv`；
- 估算 field offset；

`load_dataset_batch_for_frame(npz_path, cfg=None) -> DatasetContext`

- [fsglib/common/io.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/io.py:372)
- 根据单帧路径回溯。

### 6.2 预处理

`preprocess_frame(raw, calib, cfg) -> PreprocessedFrame`

- [fsglib/preprocess/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/preprocess/pipeline.py:4)

实现：

- 有限值掩膜、全局背景估计和全局噪声估计；
- `calib` 预留；
- 李洋后续在这里加入暗场，需保持接口不变——chenxu。

### 6.3 星点提取

`extract_stars(frame, cfg) -> list[StarCandidate]`

- [fsglib/extract/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/extract/pipeline.py:90)

质心方法：

- `weighted_centroid`
- `fixed_window_first_moment`
- `full_window_first_moment`

配置入口：

- `extract.seed_threshold_sigma`
- `extract.min_area / max_area`
- `extract.centroid_method`
- `extract.centroid_window.size`
- `extract.reject_edge_margin`
- `extract.bias_correction.*`

其他：

- ~~bias correction 的入口在 [fsglib/extract/bias.py](/home/cxgao/ET/FSG/fsglib/fsglib/extract/bias.py:69)~~（已经弃用——chenxu）。

### 6.4 候选星转观测向量

`candidates_to_observed(candidates, projector, cfg) -> list[ObservedStar]`

- [fsglib/pipeline/convert.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/convert.py:4)

接口用于通用投影链路，要求 `projector` ：

- `pixel_to_los_body(detector_id, x, y)`

**当前导星直接调用 `et_focalplane` 做像点到 LOS 的转换。**

### 6.5 参考星构造

`build_reference_stars(ctx, catalog_provider, projector, cfg) -> list[ReferenceStar]`

- [fsglib/ephemeris/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/ephemeris/pipeline.py:51)

行为：

- `init` 模式下按 boresight 查询区域星表；

- `tracking` 模式下查询跟踪目标；

- 再调用 `projector.project_to_detectors()` 生成 `predicted_xy`。
  
  `projector` 的要求：

- `project_to_detectors(los_inertial, attitude_q)`

### 6.6 匹配

`match_stars(ctx, reference_stars, cfg) -> MatchingResult`

- [fsglib/match/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/match/pipeline.py:120)

策略：

- 基于预测像点的最近邻；
- `match.algorithm` 为 `triangle` 或 `local_triangle`，再尝试三角匹配（TODO: 实现有问题，需检查——chenxu）；
- 二者中选匹配数更多的一组。

`associate_nearest(...) -> MatchingResult`

- [fsglib/match/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/match/pipeline.py:5)
- 当前主链路。

`validate_match_hypothesis(...) -> tuple[bool, dict]`

- [fsglib/match/pipeline.py](/home/cxgao/ET/FSG/fsglib/fsglib/match/pipeline.py:166)
- 做姿态跳变和残差门限检查。

### 6.7 姿态解算

`solve_attitude(solve_input, cfg) -> AttitudeSolution`

- [fsglib/attitude/solver.py](/home/cxgao/ET/FSG/fsglib/fsglib/attitude/solver.py:197)

流程：

- `solve_quest`
- `reject_outliers`
- 重新求解
- 质量标记与降级等级判定

约定：

- 四元数采用标量 `[w, x, y, z]`；
- `q_ib` / `c_ib` 表示惯性系到本体系；
- `quality_flag` 当前主要取 `VALID`、`DEGRADED`、`LOST`、`INVALID`；
- `degraded_level` 由有效 detector 数量给出。

### 6.8 评估调试

`evaluate_frame_result(...) -> FrameEvaluation | None`

- [fsglib/pipeline/evaluate.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/evaluate.py:147)
- 有 truth 时返回单帧评估，否则返回 `None`。

`summarize_sequence_result(sequence_result) -> dict`

- [fsglib/pipeline/evaluate.py](/home/cxgao/ET/FSG/fsglib/fsglib/pipeline/evaluate.py:242)
- 汇总序列指标。

`save_debug_bundle(result, cfg) -> Path | None`

- [fsglib/common/debug.py](/home/cxgao/ET/FSG/fsglib/fsglib/common/debug.py:772)
- 单帧结果；

## 7. 导星链路

### 7.1 `run_guide_first_frame_init`

1. 通过 `_load_et_coord()` 动态加载 `et_coord` 对象。
2. 按 guide detector 建立 sim 像点到 `et_focalplane` detector 像点的映射。
3. 从每个 batch 的首帧图像提取候选星。
4. 候选星经 `et_focalplane` 几何模型转成 `ObservedStar`。
5. 用 `query_detector_sources()` 为每个 detector 构造参考星。
6. 统一做匹配和 QUEST 解算。
7. 生成 `guide_error_audit`。

依赖 `et_focalplane` 接口：

- `load_registry`
- `Transformer`
- `GaiaCatalog`
- `GaiaSourceFilter`
- `query_detector_sources`

### 7.2 `run_guide_first_frame_truth_noise`

- 不从图像提取候选星；
- 先取 truth detector 像点；
- 在 detector 像素平面注入高斯噪声；
- 再进入统一的几何转换、匹配和姿态解算。

## 8. 配置

### 8.1 通用链路

 [configs/base.yaml](/home/cxgao/ET/FSG/fsglib/configs/base.yaml:1)。

- `project`
- `dataset`
- `preprocess`
- `extract`
- `match`
- `tracking`
- `ephemeris`
- `attitude`
- `evaluation`
- `logging`

### 8.2 附加配置

`run_guide_first_frame_init` ：

- [configs/guide_v1_noise_psf_etcoord.yaml](/home/cxgao/ET/FSG/fsglib/configs/guide_v1_noise_psf_etcoord.yaml:1)

`run_guide_first_frame_truth_noise` ：

- [configs/guide_truth_noise_0065pix_exact_etcoord.yaml](/home/cxgao/ET/FSG/fsglib/configs/guide_truth_noise_0065pix_exact_etcoord.yaml:1)
- 当前主链路

## 9. 结果调试

### 9.1 `FrameResult`

结构是 `FrameResult`。

字段：

- `raw`
- `preprocessed`
- `candidates`
- `observed`
- `reference`
- `matching`
- `solution`
- `evaluation`
- `meta`

### 9.2 首帧返回值

首帧返回 `dict`

### 9.3 Debug bundle

 `project.save_debug=true`，`save_debug_bundle()` 在 `project.output_dir` 下生成调试文件：

- `truth_stars.json`
- `reference_stars.json`
- `candidates.json`
- `matches.json`
- `solution.json`
- `analysis.json`
- `centroid_step_audit.json`
