"""Causal XFeat CPU ablation using an explicit local upstream checkout.

No model code or weights are downloaded implicitly. Install upstream runtime
dependencies separately. Native camera depth supplies every metric measurement.
The standard mutual ratio matcher is retained as a controlled adapter ablation,
not a reproduction of the paper's matching or SLAM benchmarks.
"""

import argparse
import importlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.datasets.tum import TUMConfig
from scene_recall.features import RGBDFeatures, sample_native_depth
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.run_metadata import sha256
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.runner import run_online


class XFeatExtractor:
    """Single-family, bounded CPU inference with recorded code and weight hashes."""

    def __init__(self, checkout, *, top_k=1024):
        import torch

        checkout = Path(checkout).resolve()
        weights = checkout / "weights" / "xfeat.pt"
        files = sorted((checkout / "modules").rglob("*.py"))
        if not files or not weights.is_file():
            raise ValueError(
                "checkout must contain upstream modules and weights/xfeat.pt"
            )
        if "modules" in sys.modules:
            raise ValueError(
                "upstream modules namespace already loaded, use a fresh process"
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(checkout))
        try:
            module = importlib.import_module("modules.xfeat")
        finally:
            sys.path.pop(0)
        if Path(module.__file__).resolve() != checkout / "modules" / "xfeat.py":
            raise ValueError("unexpected XFeat module origin")
        self.model = module.XFeat(weights=None, top_k=top_k)
        self.model.dev = torch.device("cpu")
        self.model.net.to("cpu")
        self.model.net.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        self.provenance = {
            "name": "xfeat",
            "checkout": str(checkout),
            "source_sha256": {str(p.relative_to(checkout)): sha256(p) for p in files},
            "weights_sha256": sha256(weights),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "top_k": top_k,
            "detection_threshold": 0.05,
            "association": "existing mutual L2 ratio checks",
            "metric_origin": "native observed RGB-D only",
        }

    def __call__(self, observation, calibration):
        result = self.model.detectAndCompute(observation.rgb)[0]
        pixels = result["keypoints"].cpu().numpy().astype(np.float64)
        descriptors = result["descriptors"].cpu().numpy().astype(np.float32)
        return RGBDFeatures(
            pixels,
            descriptors,
            sample_native_depth(observation, calibration, pixels),
            "xfeat",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("tum_rgbd", "eth3d_slam", "scannet_v2"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--max-reference-gap-s", type=float, default=0.05)
    parser.add_argument("--flow-recovery", action="store_true")
    parser.add_argument("--intervention", choices=("rgb", "depth"))
    parser.add_argument("--outage", type=int, nargs=2)
    args = parser.parse_args()
    if bool(args.intervention) != bool(args.outage):
        parser.error("intervention and outage must be supplied together")
    extractor = XFeatExtractor(args.checkout)
    run_online(
        args.source,
        args.output_dir,
        dataset=args.dataset,
        start=args.start,
        stop=args.stop,
        config=replace(OnlineConfig(), flow_recovery=args.flow_recovery),
        mapping=MappingConfig(pixel_stride=8),
        association=TUMConfig(max_ground_truth_difference_s=args.max_reference_gap_s),
        feature_extractor=extractor,
        invocation=sys.argv,
        intervention=(args.intervention, *args.outage) if args.outage else None,
    )


if __name__ == "__main__":
    main()
