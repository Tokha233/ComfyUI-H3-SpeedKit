"""Recompute headline metrics from recorded samples; does not run a GPU."""

from pathlib import Path
from statistics import mean
import json


def summarize(pipeline, quality):
    measurements = pipeline["measurements"]
    sums = {arm: {} for arm in ("stock", "r85")}
    for field in ("total_seconds", "dit_seconds", "video_seconds"):
        for arm in sums:
            sums[arm][field] = sum(
                mean(row[field] for row in measurements
                     if row["case"] == case["case"] and row["mode"] == arm)
                for case in pipeline["cases"]
            )
    baseline, optimized = sums["stock"], sums["r85"]
    decode = {
        arm: sum(mean(row["seconds"] for row in quality["measurements"]
                      if row["case"] == case["case"] and row["arm"] == arm)
                 for case in quality["cases"])
        for arm in ("A", "B")
    }
    return {
        "source": "historical recorded samples; not a new GPU benchmark",
        "baseline_kitchen": "0.2.35",
        "baseline_video_vae": "FP16",
        "formal_full_requests": len(measurements),
        "formal_decode_runs": len(quality["measurements"]),
        "total_latency_reduction_percent": 100 * (1 - optimized["total_seconds"] / baseline["total_seconds"]),
        "dit_latency_reduction_percent": 100 * (1 - optimized["dit_seconds"] / baseline["dit_seconds"]),
        "serial_capacity_equivalent_percent": 100 * (baseline["total_seconds"] / optimized["total_seconds"] - 1),
        "measured_service_throughput_increase": None,
        "vae_15_clip_reduction_percent": 100 * (1 - decode["B"] / decode["A"]),
        "raw_rgb_psnr_mean_db": mean(c["raw"]["psnr_db"] for c in quality["cases"]),
        "mp4_ssim_mean": mean(c["metrics"]["pixels"]["ssim"] for c in quality["cases"]),
        "lpips_mean": mean(c["metrics"]["lpips"]["mean"] for c in quality["cases"]),
        "pcm_equal_clips": sum(c["pcm_equal"] for c in quality["cases"]),
        "packaged_acceleration_validated": False,
        "kitchen_0_2_36_comparison_available": False,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pipeline = json.loads((root / "evidence/native-fp16-vs-r85.json").read_text())
    quality = json.loads((root / "evidence/native-fp16-vae-quality.json").read_text())
    print(json.dumps(summarize(pipeline, quality), ensure_ascii=False, indent=2))
