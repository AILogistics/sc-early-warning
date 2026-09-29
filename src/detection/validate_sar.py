"""在 HRSID/SSDD 上验证 AMFDet 检测精度 + HRSID→SSDD 跨域测试。

Task 2.2 / Exp 1: SAR detection validation.
跨域测试回应 AMFDet reviewer 3 的批评("缺跨数据集验证")。

设计说明:
  AMFDet 是 HRSID-trained,因此 SSDD val 本身就是 HRSID→SSDD 零样本跨域测试。
  不需要单独的第三次 val() 调用 — SSDD 结果直接标记为 cross-domain。
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

from .model import load_detector

# model.py 路径: sc_early_warning/src/detection/validate_sar.py
#   parents[0]=src/detection/, [1]=src/, [2]=sc_early_warning/, [3]=父项目根
PROJ_ROOT = Path(__file__).resolve().parents[2]
PARENT_PROJ = Path(__file__).resolve().parents[3]
HRSID_YAML = PARENT_PROJ / "HRSID.yaml"
SSDD_YAML = PARENT_PROJ / "SSDD.yaml"
EXP_OUT = PROJ_ROOT / "experiments" / "exp1_detection"


def validate(dataset_yaml: Path, weights=None, imgsz: int = 640, batch: int = 16) -> dict:
    """在数据集上验证 AMFDet,返回核心指标。"""
    model = load_detector(weights)
    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=imgsz,
        batch=batch,
        device=0,
        save_json=False,
        verbose=True,
        project=str(EXP_OUT),
        name=f"val_{dataset_yaml.stem}",
        exist_ok=True,
    )
    return {
        "dataset": dataset_yaml.stem,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. HRSID in-domain validation
    if HRSID_YAML.exists():
        print("\n=== HRSID in-domain validation ===", file=sys.stderr)
        try:
            r = validate(HRSID_YAML)
            r["experiment_type"] = "HRSID_in_domain"
            results.append(r)
            print(f"  HRSID: mAP50={r['mAP50']:.4f}, mAP50-95={r['mAP50_95']:.4f}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] HRSID val failed: {e}", file=sys.stderr)
            results.append({"dataset": "HRSID", "error": str(e)})
    else:
        print(f"[WARN] HRSID.yaml not found at {HRSID_YAML}", file=sys.stderr)

    # 2. SSDD = HRSID->SSDD cross-domain zero-shot (AMFDet is HRSID-trained)
    if SSDD_YAML.exists():
        print("\n=== SSDD (HRSID->SSDD zero-shot cross-domain) ===", file=sys.stderr)
        try:
            r = validate(SSDD_YAML)
            r["experiment_type"] = "HRSID_to_SSDD_zero_shot"
            results.append(r)
            print(f"  SSDD: mAP50={r['mAP50']:.4f}, mAP50-95={r['mAP50_95']:.4f}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] SSDD val failed: {e}", file=sys.stderr)
            results.append({"dataset": "SSDD", "error": str(e)})

    out = EXP_OUT / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Results saved to {out}", file=sys.stderr)

    # 打印总结
    print("\n" + "=" * 60, file=sys.stderr)
    print("TASK 2.2 / EXP 1 SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for r in results:
        if "error" in r:
            print(f"  {r['dataset']}: ERROR ({r['error'][:60]})", file=sys.stderr)
        else:
            tag = r.get("experiment_type", "in-domain")
            print(f"  {r['dataset']} [{tag}]: mAP50={r['mAP50']:.4f}, "
                  f"mAP50-95={r['mAP50_95']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}",
                  file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
