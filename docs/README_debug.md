# fsglib Debug Outputs

本文档说明当前 `fsglib` 会写出的主要调试产物。不同入口的输出粒度不完全
相同：compact examples 写少量 top-level JSON；exact/full-bundle examples
会写完整目录树。

## 1. 输出目录规则

默认调试目录来自：

```yaml
project:
  output_dir: outputs/debug
```

guide-specific output helper 的规则：

- 如果 `project.output_dir` 保持默认值，并且 `guide_init.dataset_root`
  存在，则输出到 `<dataset_root>_fsg-results/frameXXXXXX/`；
- `debug/` 保存 JSON、数组和审计数据；
- `figures/` 保存 matching overlays；
- 如果显式设置了 `project.output_dir`，则直接使用该目录。

truth-noise exact 和 full parallel examples 会把每一帧写到独立输出目录，并
额外保存 config snapshot 和 per-frame status。

## 2. Compact run-level 文件

常见 compact examples 会写：

- `*_result.json`: 主结果摘要。
- `*_error_audit.json`: guide error audit。
- `*_error_budget.json`: detector-to-attitude error-budget ledger。
- `*_error_budget_terms.csv`: ledger 的表格化版本。

这些文件通常位于 `outputs/debug/`，或由脚本显式设置的目录。

## 3. Debug bundle 文件

`save_debug_bundle()` 或 full-bundle examples 可能写出以下文件。实际存在与否
取决于入口、配置和是否有相应中间数据。

### 图像与候选星

- `raw.npy`: 原始图像数组。
- `preprocessed.npy`: 预处理后的图像数组。
- `background.npy`: 背景估计图或标量展开。
- `noise_map.npy`: 噪声图。
- `variance_map.npy`: 方差图。
- `valid_mask.npy`: 有效像素 mask。
- `truth_stars.json`: truth 星表，已转换到像素坐标。
- `candidates.json`: 提取候选星。
- `centroid_step_audit.json`: 可选 centroid step audit。

### 参考星与匹配

- `reference_stars.json`: 当前帧参考星和 predicted detector position。
- `observed_stars.json`: 当前帧观测星。
- `matches.json`: 最终参与姿态解算的匹配对。
- `matching/matching_result.json`: full-bundle matching 摘要。
- `matching/matched_stars.json`: full-bundle matched stars。
- `matching/observed_stars.json`: full-bundle observed stars。
- `matching/reference_stars.json`: full-bundle reference stars。
- `matching/detector_stats.json`: detector-level 统计。

### 姿态与验证

- `solution.json`: 本帧姿态解和质量摘要。
- `attitude/solution_summary.json`: 姿态解摘要。
- `attitude/covariance.json`: 小角姿态 covariance 和控制质量指标。
- `attitude/robust_rejection.json`: 姿态鲁棒剔除逐轮审计。
- `analysis.json`: 单帧误差分析。
- `audit/guide_error_audit.json`: full-bundle guide error audit。
- `validation/error_budget.json`: error-budget ledger。
- `validation/error_budget_terms.csv`: ledger CSV。

### 几何、配置和图

- `geometry/geometry_adapter.json`: exact focal-plane adapter metadata。
- `geometry/sim_to_detector_map.json`: sim pixel 到 detector pixel 的映射。
- `config/base.yaml`: base config snapshot。
- `config/<overlay>.yaml`: workflow overlay snapshot。
- `config/merged_config.yaml`: 运行时合并配置。
- `config/run_meta.json`: 运行元数据。
- `overlay_truth_candidates.png`: truth 与候选星叠加图。
- `matched_truth_bias.png`: truth 到观测质心的偏差矢量图。
- `matched_prediction_overlay.png`: predicted position 到观测质心的残差图。
- `figures/matching_overlay_summary.json`: matching overlay 输出摘要。

## 4. `solution.json` 常用字段

- `valid`: 姿态解是否通过当前质量门限。
- `num_matched`: 进入姿态求解的匹配星数量。
- `num_rejected`: 姿态鲁棒剔除数量。
- `q_ib`: scalar-first quaternion `[w, x, y, z]`。
- `covariance_rad2`: 小角姿态 covariance，单位 rad^2；缺少 matched-star
  sigma 时为 null。
- `sigma_non_roll_arcsec`: 光轴指向二维 1-sigma 不确定度。
- `sigma_roll_arcsec`: 绕 body `+Z` 的 roll 1-sigma 不确定度。
- `attitude_condition_number`: covariance normal matrix 条件数。
- `residual_rms_arcsec`: 姿态解算后 LOS residual RMS。
- `residual_max_arcsec`: 姿态解算后最大 LOS residual。
- `quality_flag`: `VALID`、`DEGRADED`、`LOST` 或 `INVALID` 等质量标签。
- `degraded_level`: 当前降级等级。
- `active_detector_ids`: 当前参与解算的 detector ids。
- `solver_iterations`: 求解和剔除迭代次数。
- `quality.meta.attitude_covariance`: covariance 可用性和 provenance。
- `quality.meta.robust_rejection`: outlier rejection 审计。
- `error_budget`: error-budget ledger 的 JSON 副本。
- `timings_s`: 运行耗时信息。

## 5. `matching.debug` 常用字段

- `algorithm`: 请求的 matching algorithm。
- `selected_strategy`: 实际采用的策略。
- `success`: matching 是否达到支持和残差要求。
- `num_candidate_edges`: predicted-position 候选边数量。
- `num_predicted_position_matches`: predicted-position 匹配数量。
- `num_local_pyramid_matches`: local-pyramid 匹配数量。
- `mean_residual_pix`: predicted detector position 到观测质心的平均残差。
- `rms_residual_pix`: detector pixel residual RMS。
- `unique_assignment_enabled`: 是否执行一对一 assignment。
- `num_unique_matches`: 一对一 assignment 后保留的匹配数量。
- `pyramid_debug`: local-pyramid seed、expansion 和 rejection 细节。
- `nearest_vs_pyramid`: predicted-position 与 local-pyramid 对比。

## 6. 评估字段

`FrameEvaluation` 或 result payload 中常见字段：

- `evaluation.centroid_mae_pix`
- `evaluation.non_roll_error_arcsec`
- `evaluation.roll_error_arcsec`
- `evaluation.total_attitude_error_arcsec`
- `evaluation.error_budget`

这些字段依赖 truth、matched-star covariance 和配置开关；缺少输入时会保持
null 或不生成。
