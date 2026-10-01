# SceneRecall

Persistent open-vocabulary 3D scene memory for robotic perception.

## Status

Work in progress.

## ScanNet frame extraction

Load one original frame from a processed ScanNet v2 capture:

```python
from scene_recall.datasets.scannet import load_scannet_frame

sequence = load_scannet_frame(
    "~/datasets/scannet/scans/scene0000_00/scene0000_00.sens",
    original_frame_index=0,
)
observation = sequence.observations[0]
```

The adapter supports version 4 with JPEG color and zlib uint16 depth. It verifies
the expected calibrated StructureSensor registration, preserves the native image
grids, and returns depth in meters with NaN for invalid measurements. Preceding
frame payloads are skipped. The original index and raw timestamps are recorded in
provenance, while the one-observation sequence has canonical frame index 0.
Zero depth timestamps are unavailable. Stored reference poses use canonical
validation without repair, with the all-`-inf` tracking sentinel represented by
`None`. See `load_scannet_frame`'s docstring for supported calibration and errors.
