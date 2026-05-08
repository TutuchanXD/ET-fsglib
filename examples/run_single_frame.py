"""
主探测器单帧解算示例。

用途：
- 读取主探测器单张仿真 npz；
- 按单帧初始化链路完成候选提取、参考星匹配和姿态解算；
- 生成调试 bundle，并在终端打印单帧解算结果。

使用配置：
- configs/base.yaml
- configs/main_sim_v2.yaml

默认输入：
- /home/cxgao/Results/FSG_images_sims_legacy_20260508/v2/batch0_ra304.0980_dec51.4330/frames/scope0_coadd_000000_000000.npz
"""

import sys
from pathlib import Path
import yaml

# Add the parent directory of fsglib to python path so we can run this directly
sys.path.append(str(Path(__file__).parent.parent))

from fsglib.pipeline.run_init import run_single_frame_init
from fsglib.common.io import load_dataset_batch_for_frame
from fsglib.common.debug import save_debug_bundle
from fsglib.models.mock import build_models


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    cfg_path = Path("configs/base.yaml")
    if not cfg_path.exists():
        print(f"Error: Config file not found at {cfg_path.absolute()}")
        return

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for extra_cfg_path in [Path("configs/main_sim_v2.yaml")]:
        if extra_cfg_path.exists():
            cfg_extra = yaml.safe_load(extra_cfg_path.read_text(encoding="utf-8"))
            cfg = _deep_update(cfg, cfg_extra)

    # Use one of the provided batch npz files
    npz_path = "/home/cxgao/Results/FSG_images_sims_legacy_20260508/v2/batch0_ra304.0980_dec51.4330/frames/scope0_coadd_000000_000000.npz"
    if not Path(npz_path).exists():
        print(f"Error: NPZ file not found at {npz_path}")
        return

    dataset_ctx = load_dataset_batch_for_frame(npz_path, cfg=cfg)
    models = build_models(cfg)
    result = run_single_frame_init(
        npz_path=npz_path, cfg=cfg, models=models, dataset_ctx=dataset_ctx
    )

    bundle_dir = save_debug_bundle(result, cfg)
    print("----------------------------------------")
    print("Single Frame Pipeline Execution Results:")
    print("----------------------------------------")
    print(f"Attitude valid: {result.solution.valid}")
    print(f"Matched stars:  {result.solution.num_matched}")
    print(f"Residual RMS (arcsec): {result.solution.residual_rms_arcsec:.2f}")
    if result.evaluation is not None:
        print(f"Centroid MAE (pix): {result.evaluation.centroid_mae_pix}")
        print(f"Boresight error (arcsec): {result.evaluation.boresight_error_arcsec}")
        print(f"Non-roll error (arcsec): {result.evaluation.non_roll_error_arcsec}")
        print(f"Roll error (arcsec): {result.evaluation.roll_error_arcsec}")
        print(
            f"Total attitude error (arcsec): {result.evaluation.total_attitude_error_arcsec}"
        )
    if result.solution.valid:
        print(f"Quaternion [w, x, y, z]: {result.solution.q_ib}")
    if bundle_dir is not None:
        print(f"Debug bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
