# fsglib API Notes

本文档记录当前 `fsglib` 维护中的主要 API、数据结构和运行入口。详细
YAML key 说明见 `docs/yaml_configuration_reference.md`，调试产物说明见
`docs/README_debug.md`。

## 1. 当前维护入口

主要维护入口：

- `fsglib.pipeline.run_guide_first_frame_init(cfg)`
- `fsglib.pipeline.run_guide_first_frame_truth_noise(cfg)`
- `fsglib.pipeline.run_init.run_single_frame_init(npz_path, cfg, models, dataset_ctx=None)`
- `fsglib.pipeline.run_tracking.run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=None)`

其中前两个是 ET 四 guide detector 联合验证的主路径。后两个是通用单帧
和序列接口，需要外部注入 `models["projector"]`、`models["catalog"]`，
以及在 lost-in-space 路径中可选的 `models["lis_index"]`。

## 2. 链路分层

### 数据输入层

位置：

- `fsglib/common/io.py`

职责：

- 读取单帧 NPZ；
- 回溯或读取 batch 目录；
- 构造 `DatasetContext`；
- 在可用时解析帧内 truth payload 和 `stars.ecsv` 静态 truth。

### 图像到观测向量

位置：

- `fsglib/preprocess/calibration.py`
- `fsglib/preprocess/pipeline.py`
- `fsglib/extract/pipeline.py`
- `fsglib/pipeline/convert.py`
- `fsglib/ephemeris/guide_geometry.py`
- `fsglib/pipeline/run_guide_init.py`
- `fsglib/pipeline/run_guide_truth_noise.py`

职责：

- `RawFrame -> PreprocessedFrame -> StarCandidate -> ObservedStar`；
- 对真实图像执行 detector calibration、背景/方差估计和候选星提取；
- 对 truth-noise 工作流直接从 truth detector 坐标构造合成观测；
- 使用 exact `et_focalplane` adapter 生成 body-frame LOS。

### 参考星、匹配与姿态层

位置：

- `fsglib/ephemeris/pipeline.py`
- `fsglib/match/pipeline.py`
- `fsglib/match/pyramid.py`
- `fsglib/match/lost_in_space.py`
- `fsglib/attitude/solver.py`

职责：

- 构造 `ReferenceStar`；
- 执行 predicted-position、local-pyramid、reacquire 或 lost-in-space 匹配；
- 用 QUEST 求解惯性系到本体系姿态；
- 执行姿态 covariance 估计和鲁棒 outlier rejection。

### 评估与输出层

位置：

- `fsglib/pipeline/evaluate.py`
- `fsglib/pipeline/guide_error_audit.py`
- `fsglib/pipeline/error_budget.py`
- `fsglib/pipeline/guide_outputs.py`
- `fsglib/common/debug.py`

职责：

- 单帧与序列指标；
- guide error audit；
- detector-to-attitude error-budget ledger；
- debug bundle 和 matching overlay 输出。

## 3. 主要入口

### `run_guide_first_frame_init(cfg) -> dict`

适用场景：

- 真实图像质心提取；
- transit 或 microlensing 四 guide detector 联合首帧初始化；
- reference stars 来自 `et_focalplane + GaiaCatalog`；
- matching 和 QUEST 均由 `fsglib` 执行。

输入要求：

- `cfg["guide_init"]` 提供 `dataset_root`、`detector_batches` 和筛选参数；
- `cfg["et_coord"]` 提供 `src_dir`、`data_dir`、`gaia_root_dir`；
- 每个 batch 目录通常包含 `frames/*.npz`、`run_meta.json`、`stars.ecsv`。

返回字典常用 key：

- `solution`
- `matching`
- `observed_count`
- `reference_count`
- `detector_stats`
- `sim_to_detector_map`
- `geometry_adapter`
- `error_audit`
- `error_budget`
- `meta`

说明：

- guide 链路只使用 exact ET focal-plane adapter；
- detector family 由 YAML 中的 `et_coord.config_factory` 选择；
- `geometry_adapter` 是当前几何模型输出字段。

### `run_guide_first_frame_truth_noise(cfg) -> dict`

适用场景：

- 不从图像提取质心；
- 从 truth detector 像点注入高斯 centroid noise；
- 隔离几何、匹配和姿态求解误差；
- 支持 compact 和 exact/full-bundle 两类 example。

配置入口：

- `cfg["guide_truth_noise"]`
- `cfg["et_coord"]`
- `cfg["match"]`
- `cfg["attitude"]`

返回字典在 guide init 基础上额外包含：

- `synthetic_centroid_model`
- `debug_context`，当调用方请求 `include_debug_context=True` 时存在。

### `run_single_frame_init(...) -> FrameResult`

适用场景：

- 通用单帧初始化；
- 测试或非 guide-specific 投影模型。

`models` 要求：

- `models["projector"]`
- `models["catalog"]`
- 可选 `models["calib"]`

返回：

- `FrameResult`

### `run_sequence_tracking(...) -> SequenceResult`

适用场景：

- 通用序列跟踪；
- 状态机在 `init_known_field`、`tracking`、`local_reacquire`、
  `lost_in_space`、`safe_lost` 之间转移；
- `lost_in_space` 模式使用 `LostInSpaceMatcher`，需要运行时提供
  `models["lis_index"]`。如果缺少 index，会返回 invalid frame，并在 debug
  中记录 `lost_in_space_index_missing`。

返回：

- `SequenceResult.frames`
- `SequenceResult.mode_history`
- `SequenceResult.state_history`
- `SequenceResult.metrics`

## 4. 数据结构

定义位置：

- `fsglib/common/types.py`
- `fsglib/ephemeris/types.py`

核心结构：

- `RawFrame`: 原始图像、时间、单位和 truth metadata。
- `PreprocessedFrame`: 预处理图像、背景、噪声图、方差图、有效掩膜和
  artifact masks。
- `StarCandidate`: 提取候选星，包含 centroid、flux、SNR、bbox、shape、
  flags 和 `centroid_cov_pix`。
- `ObservedStar`: 匹配和姿态输入，包含 detector 像点、`los_body`、
  centroid covariance、LOS covariance 和 `sigma_angle_arcsec`。
- `ReferenceStar`: catalog id、`los_inertial`、各 detector 的
  `predicted_xy` 和 visibility。
- `MatchedStar`: observation-reference 配对，姿态求解使用
  `los_body` 和 `los_inertial`。
- `MatchingResult`: matched stars、success 标记和 debug metadata。
- `AttitudeSolution`: `q_ib`、`c_ib`、residuals、covariance、quality 和
  robust rejection metadata。
- `FrameResult`: 单帧通用链路结果。
- `SequenceResult`: 序列链路结果。

## 5. 预处理

`preprocess_frame(raw, calib, cfg) -> PreprocessedFrame`

当前行为：

- finite mask；
- ADC clip；
- saturation guard；
- bias、dark、FPN、flat、bad-pixel calibration chain；
- `median`、`sigma_clip_global`、`mesh_median` 背景估计；
- `empirical_robust` 或 `poisson_read_noise` 方差模型；
- unit-aware ADU/electron conversion；
- artifact masks 写入 `PreprocessedFrame.artifact_masks`；
- calibration provenance 写入 `preprocess_meta`。

`calib` 通常由 `fsglib.models.mock.build_models(cfg)` 根据
`preprocess.*_path` 加载。默认 guide 配置指向 2049x2049 fake/no-op
calibration assets，用于保持链路可运行；真实 calibration assets 应通过 YAML
覆盖。

## 6. 星点提取

`extract_stars(frame, cfg) -> list[StarCandidate]`

支持的 centroid method：

- `weighted_centroid`
- `adaptive_moment_centroid`
- `fixed_window_first_moment`
- `full_window_first_moment`

保留但未实现的接口：

- `psf_template_fit`，需要 `psf.template_bundle_path`，当前会显式报错。

当前行为：

- SNR 图上 seed/grow hysteresis segmentation；
- 8-connected grown component；
- shape metrics：ellipticity、FWHM proxy、sharpness；
- edge、area、artifact overlap、degenerate source 过滤；
- multi-peak blend detection，可 `flag_only` 或 `reject`；
- `centroid_cov_pix` 和 `centroid_sigma_*` flags；
- `candidates_to_observed()` 可把 centroid covariance 传播到 LOS/angular sigma。

## 7. 匹配

`match_stars(ctx, reference_stars, cfg) -> MatchingResult`

`match.algorithm` 当前活跃值：

- `predicted_position`
- `local_pyramid`
- `predicted_position_and_local_pyramid`
- `predicted_position_with_pyramid_reacquire`

deprecated compatibility values：

- `triangle`
- `local_triangle`

predicted-position matcher：

- 按 detector predicted pixel residual 构造候选边；
- 默认启用一对一 assignment；
- debug 中记录 candidate edge 数、mean/RMS residual 等。

local-pyramid matcher：

- 使用当前 `ReferenceStar` 列表，不依赖旧 GSC index；
- 支持 single-detector 和 mixed-detector seeds；
- 支持 pair/query cache；
- reacquire 可使用 seed-attitude-only geometry expansion；
- debug 中记录 seed、expansion、ambiguity 和 per-detector residual audit。

lost-in-space matcher：

- 位于 `fsglib.match.lost_in_space`；
- 使用预构建 LIS index；
- 跟踪状态机在 `lost_in_space` 模式中调用；
- index 可通过 `python -m fsglib.tools.build_lis_index ...` 构建。

## 8. 姿态解算

`solve_attitude(solve_input, cfg) -> AttitudeSolution`

约定：

- `C_ib` 把 inertial vector 映射到 body vector；
- `q_ib` 是同一旋转的 scalar-first quaternion `[w, x, y, z]`；
- SciPy quaternion 顺序转换必须使用 solver helper。

当前行为：

- QUEST 求解；
- 支持 `snr`、`centroid_variance`、`variance_snr_hybrid` 权重；
- 从 matched-star `sigma_angle_arcsec` 估计小角姿态 covariance；
- 输出 `covariance_rad2`、`sigma_non_roll_arcsec`、
  `sigma_roll_arcsec`、`attitude_condition_number`；
- iterative robust rejection 记录在
  `quality["meta"]["robust_rejection"]`；
- 若缺失 sigma，仍可求解姿态，但 covariance metadata 会标记 unavailable。

## 9. 评估与审计

`evaluate_frame_result(...) -> FrameEvaluation | None`

- 有 truth 时输出 centroid、roll/non-roll/total attitude metrics；
- 缺少 truth 时返回 `None`；
- 可挂载 frame-level error-budget summary。

`summarize_sequence_result(sequence_result) -> dict`

- 汇总 valid frame、mode history、runtime 和 error budget 聚合指标。

`compute_guide_error_audit(...) -> dict`

- 比较 truth、extracted centroid、predicted detector position、LOS geometry、
  matching 和 final attitude；
- 用于定位误差来源。

`build_error_budget_ledger(...) -> ErrorBudgetLedger`

- 输出 detector/preprocess/centroid/matching/attitude 的结构化 ledger；
- 每个 term 包含单位、来源、假设、available 状态和 unavailable reason；
- 缺少物理输入时记录 unavailable，不用 0 伪装未知误差；
- fake/no-op calibration assets 会在 provenance/assumption 中显式标记。

## 10. 常用 examples

```bash
python examples/run_guide_first_frame.py
python examples/run_microlens_guide_first_frame.py
python examples/run_guide_first_frame_truth_noise.py
python examples/run_guide_first_frame_truth_noise_exact.py
python examples/run_transit_full_truth_noise_exact_parallel.py
python examples/run_pr21_error_budget_smoke.py
python examples/run_single_frame.py
```

## 11. Debug 输出

`project.save_debug=true` 时，`save_debug_bundle()` 会在
`project.output_dir` 下生成 debug bundle。guide-specific examples 还会写出：

- top-level result/audit/error-budget JSON；
- matching records；
- geometry metadata；
- config snapshots；
- detector-level debug records；
- matching overlays。

具体文件布局见 `docs/README_debug.md`。
