# 调试输出

## 1. 输出目录

单帧调试在：

- `/home/cxgao/ET/FSG/fsglib/outputs/debug`

同名 bundle：

- `/home/cxgao/ET/FSG/fsglib/outputs/debug/scope0_coadd_000000_000000`

## 2.  bundle 中的文件

- `raw.npy`
  - 原始图像数组。
- `preprocessed.npy`
  - 预处理后的图像数组。
- `noise_map.npy`
  - 预处理阶段估计的噪声图。
- `truth_stars.json`
  - 来自 `stars.ecsv` 的静态真值星表，已转换到像素坐标。
- `reference_stars.json`
  - 当前帧构造出来的参考星，以及其预测像点。
- `candidates.json`
  - 星点提取阶段输出的候选质心。
- `matches.json`
  - 最终参与姿态解算的匹配对，包含 truth / predicted / observed 三类信息。
- `solution.json`
  - 本帧主结果。
- `analysis.json`
  - 对误差来源做进一步分解后的结果。
- `overlay_truth_candidates.png`
  - 静态 truth 与提取质心叠加图。
- `matched_truth_bias.png`
  - matched 星从静态 truth 指向观测质心的偏差矢量图。
- `matched_prediction_overlay.png`
  - 参考预测像点指向观测质心的残差矢量图。
- `README.md`
  - 当前 bundle 的快速说明。

## 3. `solution.json` 字段

- `valid`
  - 本帧姿态解是否通过当前门限。
- `num_matched`
  - 进入姿态解算的匹配星数。
- `num_rejected`
  - 姿态求解阶段剔除的星数。
- `q_ib`
  - 四元数 `[w, x, y, z]`，表示惯性系到本体系的旋转。
- `residual_rms_arcsec`
  - 姿态解算后，matched 星方向矢量残差的 RMS，单位角秒。
- `residual_max_arcsec`
  - 姿态解算后最大方向残差，单位角秒。
- `quality_flag`
  - 当前质量标签。
- `degraded_level`
  - 当前解算的降级等级。
- `active_detector_ids`
  - 当前参与解算的探测器编号。
- `solver_iterations`
  - 求解器迭代次数。
- `quality`
  - 姿态解算的质量摘要，如输入星数、使用星数、残差门限。
- `timings_s`
  - 各阶段耗时。
- `matching.debug.mean_residual_pix`
  - 匹配阶段中，预测像点到观测质心的平均像面残差。
- `evaluation.centroid_mae_pix`
  - 提取质心到静态 `stars.ecsv` 的平均最近邻距离。
- `evaluation.non_roll_error_arcsec`
  - 非绕光轴姿态误差。
- `evaluation.roll_error_arcsec`
  - 绕光轴姿态误差。
- `evaluation.total_attitude_error_arcsec`
  - 总姿态误差角。
