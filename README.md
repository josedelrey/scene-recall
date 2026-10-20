# SceneRecall

SceneRecall aims to provide persistent open-vocabulary 3D scene memory for
robotic perception. The current implementation provides canonical RGB-D readers,
segmented odometry, persistent multiview feature tracks, colored surface-voxel
mapping and held-out fusion validation.
Semantic memory and retrieval remain future work.

## Setup and first run

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). From the repository
root, install the locked environment:

```bash
uv sync --locked
```

Supply a processed ScanNet v2 `.sens` capture or an extracted TUM RGB-D sequence.
Keep datasets outside Git, using the ignored `data/` directory or an external
path. Store generated artifacts in ignored `outputs/`. Commands run from the
repository root and use fresh output directories.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  scripts/rgbd_odometry.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_desk \
  --backend sparse --stop 200 --output-dir outputs/odometry/desk_sparse_200

uv run --locked python scripts/rgbd_mapping.py \
  --odometry-run outputs/odometry/desk_sparse_200 \
  --output-dir outputs/mapping/desk_sparse_200
```

Generate a self-contained Rerun recording and open it in the native viewer:

```bash
uv sync --locked --extra visualization
uv run --locked --extra visualization python scripts/rgbd_recording.py \
  --mapping-run outputs/mapping/desk_sparse_200 \
  --output outputs/visualization/desk_sparse_200.rrd
uv run --locked --extra visualization rerun outputs/visualization/desk_sparse_200.rrd
```

The recording embeds RGB-D keyframes, maps, poses, diagnostics and its viewer
layout. Opening it needs no dataset or network. NPZ, PLY and JSON remain the
scientific data formats. See [Rerun visualization](docs/visualization.md) for
image-free export, relocated runs, colored TSDF meshes and synchronized held-out
depth/error inspection across fusion experiments.

Sparse is the default backend. Tracking losses create independent segments,
and mapping preserves those boundaries. Reference poses are used for diagnostics only.

## Documentation

| Topic | Guide |
| --- | --- |
| Canonical arrays, calibration and adapter responsibilities | [RGB-D data contract](docs/data_contract.md) |
| Coordinates, transforms and projection | [Camera geometry](docs/camera_geometry.md) |
| ScanNet reading, point clouds and pair diagnostics | [ScanNet support](docs/scannet.md) |
| TUM input, calibration and timestamp association | [TUM support](docs/tum.md) |
| Backends, failure policy, exports and trajectory metrics | [RGB-D odometry](docs/odometry.md) |
| Persistent tracks, shared landmarks and covisibility | [Multiview measurements](docs/multiview_tracks.md) |
| Bounded experimental pose-only refinement and map evaluation | [Pose-only multiview experiment](docs/pose_only_multiview.md) |
| Opt-in shared-landmark refinement and matched temporal controls | [Local RGB-D BA experiment](docs/local_rgbd_ba.md) |
| Matched comparisons, analysis and repeat verification | [Odometry comparison workflow](docs/odometry_backend_comparison.md) |
| Voxel fusion, local submaps and scientific exports | [RGB-D mapping](docs/rgbd_mapping.md) |
| Native offline visualization and recordings | [Rerun workflow](docs/visualization.md) |
| Optional TSDF fusion, connected meshes and pose-bias controls | [Volumetric mapping investigation](docs/volumetric_mapping.md) |
| Withheld fusion views, rendering and paired scoring | [Held-out map validation](docs/heldout_mapping_validation.md) |

## Development checks

GitHub Actions runs these checks on Python 3.12 in separate locked base and
visualization environments. The visualization job runs the full suite with the
Rerun SDK installed. Tests use synthetic inputs and require no external dataset.
Shared test builders live in `tests/helpers/`.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Recording integration tests run when the visualization extra is installed:

```bash
uv run --locked --extra visualization pytest -q \
  tests/test_visualization.py tests/test_experiment_visualization.py
```
