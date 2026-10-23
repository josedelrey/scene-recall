# Online frontend research

This follow-up extends the [causal online system](online_slam.md) with opt-in
image-flow recovery, reduced-resolution SIFT detection, an explicit learned
feature boundary and a faster native-depth validity filter. Existing frontend
and SLAM policy defaults remain unchanged. The depth-filter implementation
preserves the existing full-neighborhood acceptance rule.

## Design and research

The previous ETH3D results motivate separating two problems. Adjacent frames can
retain trackable patches while independently detected SIFT points disappear.
Historical views require recognition across larger changes and cannot rely on
short-range optical flow. Increasing accepted poses alone does not establish
correct motion, correct submap relationships or better reconstruction.

The selected geometric path tracks new Shi–Tomasi corners with pyramidal
Lucas–Kanade flow, as described in the
[OpenCV optical-flow documentation](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).
Forward/backward image consistency proposes associations. A frozen PnP fit and
withheld image/native-depth measurements verify the motion. This provides a
complement to descriptor matching without changing the metric observation model.

The learned experiment uses
[XFeat](https://arxiv.org/abs/2404.19174), selected for its compact feature model
and CPU-oriented design. The adapter runs the official model on raw RGB and
keeps at most 1,024 64-dimensional descriptors. It retains SceneRecall's existing
mutual L2 ratio checks and geometric verification. This is a complete-pipeline
adapter ablation, not a reproduction of XFeat's published matcher or benchmark.
The feature budget and detector also differ from the uncapped SIFT control.

[LightGlue](https://arxiv.org/abs/2306.13643) offers learned adaptive association
and remains a candidate for difficult historical pairs. It is not evaluated here.
[DROID-SLAM](https://arxiv.org/abs/2108.10869) couples recurrent correspondence
updates with dense bundle adjustment.
[DPVO](https://arxiv.org/abs/2208.04726) reduces that computation through learned
patch tracking. [MASt3R-SLAM](https://arxiv.org/abs/2412.12392) integrates learned
3D reconstruction and matching priors. These motivate future joint frontend and
backend work. They have not been benchmarked against SceneRecall here, and their
published speed or accuracy figures do not describe this CPU implementation.
This stage chooses a small learned appearance model while preserving native
metric depth and the established causal state/revision contracts.

## Measurement and state contracts

`--flow-recovery` adds a fallback after established local map and pair tracking
fail. It uses only the preceding and incoming observations, at most 1,000 corners,
a 21-pixel LK window and four pyramid levels. Proposals must pass a one-pixel
forward/backward check and a photometric error threshold of 20. These are frozen
engineering gates, not calibrated match probabilities.

Only original source-camera depth creates metric anchors. Target depth is checked
where available and is never filled or predicted. The existing registration gate
fits even correspondence rows, validates odd rows without refitting them, and
checks spatial coverage and bearing information. Explicit associations must be
one-to-one. Agreement with other accepted recent-anchor proposals is required.
The accepted candidate with most validation support supplies one pose. Candidates
are not fused as independent factors. Optical-flow patches can overlap and share
image noise. Splitting rows does not remove those correlations or establish
statistical independence.

Flow associations remain pair-local measurements. They do not become invented
SIFT identities, persistent landmarks or learned depth factors. Local tracking
failure boundaries, historical loop confirmation, global optimization and
revision-aware map rebuilding retain their established contracts.

`--feature-scale 0.5` changes only SIFT's detection image. Pixels are mapped back
using pixel-center coordinates and actual resized dimensions. Original camera
intrinsics and original native depth remain authoritative. CLAHE is applied
before detection-image resizing. The reduced detector is a speed/association
trade-off and is not promoted to the default.

The XFeat adapter uses a declared descriptor family with its native dimension.
It does not pad descriptors or label learned measurements as SIFT. Its place
shortlist uses a normalized mean of signed learned descriptors rather than
RootSIFT. Descriptor votes and withheld geometric checks still decide which
proposals reach loop confirmation. This simple learned-descriptor aggregation
is not a trained place recognizer.

The external extractor boundary requires the caller to own preprocessing.
Combining it with the built-in CLAHE/scale settings is rejected. The raw runner
records the external implementation and weight hashes, framework version,
feature budget and CPU policy. It strips reference poses before extraction or
SLAM processing. Reference poses are used only in evaluation.

The native-depth filter now reduces nine aligned image slices. NaNs propagate,
borders remain rejected, all neighbors must be finite and positive, and the
maximum-minus-minimum depth threshold is unchanged. It preserves native dtype
and does not modify the input. Tests compare against the original neighborhood
contract, including holes, invalid samples and small images.

## Running the alternatives

The geometric option uses the normal environment:

```bash
uv run python scripts/rgbd_slam.py --eth3d-path local_inputs/recording \
  --output-dir outputs/flow_run --contrast clahe --feature-scale 0.5 \
  --flow-recovery --map
```

For the learned experiment, obtain the official
[XFeat implementation](https://github.com/verlab/accelerated_features) at revision
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`, retaining its original license and
`weights/xfeat.pt`. The adapter does not download or install anything. Use a
separate environment with SceneRecall and the upstream runtime requirements
(PyTorch and tqdm). The experiment used CPU PyTorch 2.14.1 and one intra/inter-op
thread. Source and model dependencies remain outside the standard project
environment and lockfile.

```bash
python experiments/xfeat_online.py --checkout local_inputs/accelerated_features \
  --dataset eth3d_slam --source local_inputs/recording \
  --output-dir outputs/xfeat_run
```

`OnlineSLAM` and `run_online` accept a `feature_extractor` object. Its call returns
`RGBDFeatures` and its `provenance` attribute records model/configuration identity
for exported runs. A fresh process is required for the experimental upstream
module loader. All model execution and native-depth sampling occur causally
inside the processing call. No precomputed features or retrospective pose replay
enter these comparisons.

## Evaluation protocol

Development used the first 120 inputs of ETH3D Sofa 3, Camera Shake 3 and TUM XYZ.
These recordings already informed earlier work. Optical-flow gates were fixed
before those runs. The selected half-resolution/flow and XFeat configurations
were frozen before independent transfer and the newly acquired ETH3D Desk 3
recording were processed. Desk 3's first 1,000 inputs form the held-out prefix.
This is held out from this development process, without a claim that every scene
is absent from the learned model's training data.

Full recordings and independent windows are evaluated through raw extraction,
tracking, historical verification, global corrections and reconstruction.
Controlled RGB/depth outages share their source recording and are not independent
replicates. Causal emitted estimates and final revised maps are scored separately.
Common root-connected populations control coverage changes. Newly recovered
local transitions are checked against reference relative motion, alongside newly
introduced failures and externally inconsistent loops. The 10 cm / 5 degree
flag is a diagnostic threshold, not proof of correctness below it.

Fixed training-view/interleaved-target map comparisons use native samples and
reference target cameras. Different connected training populations change
coverage as well as geometry. Exact refusion audits separately verify that saved
reconstruction chunks agree with the exported trajectory. They establish
internal revision consistency, not external surface accuracy.

The concurrent matrix supplies robustness evidence. Serial, separate-process
runs supply runtime comparisons. Timings include raw input, feature extraction,
tracking, place queries, graph solves and map updates, and exclude final scoring
and export. Peak RSS includes framework/model allocation. Neither causal replay
nor average throughput establishes live camera-rate scheduling.

Experimental artifacts are retained under ignored `outputs/frontend_study/`.
The directory contains frozen manifests, original and delivered input digests,
model source/weight identity, source archives, raw runs, summaries and preservation
checks. Existing scientific outputs are protected by pre-study SHA-256 records.

## Development evidence

Each row below is one 120-input prefix. Repeated policies are not independent
recordings. `fast` denotes half-resolution CLAHE SIFT plus flow recovery.

This initial prototype passed `fx=fy=0.5` to OpenCV. Before the broader frozen
matrix, resizing was made explicit in integer output dimensions so its image
resampling and pixel-coordinate mapping agree for odd widths. ETH3D images are
739 pixels wide, so the initial and corrected half-scale images differ. These
prototype rows are retained as development evidence. Current-policy results
come from the full matrix, and fresh current-policy prefixes are checked against
those full runs. Cross-version differences are not treated as causal failures
or hidden through a numerical tolerance.

| Recording | CLAHE losses | Flow losses | Half-scale losses | Fast losses | XFeat losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sofa 3 | 49 | 4 | 52 | 4 | 0 |
| Camera Shake 3 | 106 | 94 | 104 | 94 | 75 |
| XYZ | 0 | 0 | 0 | 0 | 0 |

No newly accepted transition in these comparisons exceeded the diagnostic
10 cm / 5 degree relative-motion threshold. This does not imply small accumulated
drift. XFeat's Sofa root stays connected across all 120 inputs but reaches
70.74 mm emitted translation RMSE. The flow/SIFT alternatives retain smaller
connected populations and must be compared on common frames.

The Sofa coverage-aware capped surface error is 47.29 mm for CLAHE, 34.50 mm
for fast and 56.17 mm for XFeat. Their connected training-view counts are 5, 7
and 12. With the same five training views, errors are 47.29, 42.99 and 47.68 mm.
XFeat's availability improvement therefore does not establish a map-quality gain.
Camera Shake's map improvement from 140.57 to 97.25 mm under XFeat comes from
one versus two connected training views. With one shared training view, the maps
are identical. XYZ's fixed 12-view map errors are 51.90, 52.76 and 48.24 mm for
CLAHE, fast and XFeat. All development chunks passed exact refusion.

The depth-filter microbenchmark compares the original and aligned-slice versions
on 30 shape/dtype/threshold combinations with identical outputs. On a VGA depth
image, medians of 20 calls are approximately 73.30 ms and 1.80 ms. This is an
isolated operation result, and complete-pipeline timing is reported separately.

## Complete-pipeline transfer evidence

The full matrix and held-out prefix cover 4,404 distinct natural inputs across
eight recordings. Repeated prefixes and controlled XYZ outages do not increase
that count. The following translation RMSEs use the common emitted root-connected
population of the three policies. XYZ's control is the original raw online policy.
Other controls use full-resolution CLAHE. This is a comparison of configured
pipelines, not an isolated learned-versus-handcrafted descriptor benchmark.

| Recording / inputs | Control / fast / XFeat losses | Common connected inputs | Control / fast / XFeat emitted translation RMSE |
| --- | --- | ---: | --- |
| XYZ / 792 | 0 / 0 / 0 | 792 | 24.50 / 20.90 / 27.90 mm |
| RPY / 694 | 0 / 0 / 0 | 694 | 64.78 / 117.82 / 92.16 mm |
| Pioneer / 600 | 27 / 23 / 53 | 75 | 37.05 / 30.23 / 70.81 mm |
| Walking / 300 | 0 / 0 / 0 | 300 | 165.39 / 130.39 / 65.87 mm |
| ScanNet / 200 | 0 / 0 / 0 | 200 | 78.52 / 103.44 / 79.54 mm |
| Sofa 3 / 533 | 133 / 26 / 0 | 201 | 27.27 / 31.48 / 166.29 mm |
| Camera Shake 3 / 285 | 177 / 135 / 107 | 7 | 24.50 / 15.72 / 8.66 mm |

Inputs lacking reference poses are excluded from RMSE. Small common populations,
particularly Camera Shake's seven inputs, cannot represent whole-recording
accuracy. Its largest emitted component grows from 38 inputs to 76 with fast
and 63 with XFeat, while the initial component remains only 10, 7 and 11 inputs.
Fragmentation therefore remains severe.

On full Sofa, fast rescues 115 baseline failures and introduces eight new ones.
XFeat rescues all 133 and tracks the entire recording, but its whole emitted
translation RMSE is 157.47 mm. Fast emits 262 root-connected poses and later
revises 361 into that component. The control emits 211 and later connects 278.
Those retrospective additions are not reported as causal availability.
Pioneer fast rescues nine transitions while adding five failures. XFeat rescues
six while introducing 32. No newly rescued transition in the full matrix or
held-out comparison exceeds the diagnostic 10 cm / 5 degree threshold. This
local-transition result does not exclude accumulated bias or wrong global
relationships.

RPY retains externally inconsistent active loops: 6/180 for CLAHE, 4/103 for
fast and 1/87 for XFeat exceed 10 cm / 5 degrees. Reducing that count does not
ensure improved trajectory accuracy, as the RPY errors demonstrate. Pioneer
has one such active loop under CLAHE and none under the two alternatives.

The RGB outage remains eleven failed transitions for all methods. None invents
localization through the ten blank images. Historical recognition causally
reconnects the later root population to 105 inputs for all three. The depth
outage retains zero failures under all three, with native source anchors and
image observations rather than synthesized metric samples.

Fixed-view map scores also remain mixed:

| Recording | Control / fast / XFeat capped surface error | Interpretation |
| --- | --- | --- |
| XYZ | 46.90 / 47.64 / 44.82 mm | Same 24 training views |
| RPY | 73.72 / 88.68 / 82.21 mm | Same 24 training views |
| Pioneer | 152.02 / 150.82 / 152.77 mm | Same 8 connected training views |
| ScanNet | 76.30 / 92.27 / 90.86 mm | Same 20 training views |
| Sofa 3 | 55.91 / 46.92 / 39.11 mm | Connected training counts 12 / 13 / 24 |
| Sofa 3, common training | 43.14 / 51.37 / 54.82 mm | Same 11 training views |
| Camera Shake 3 | 151.12 / 151.12 / 93.90 mm | Connected training counts 1 / 1 / 2 |

Camera Shake's common one-view maps are identical. Sofa's full-map gains include
coverage and reverse under the common-training comparison. These results argue
against promoting either alternative based on tracking counts alone. All saved
chunks in the full matrix and held-out run passed exact refusion, including
runs whose strict external map score is withheld for missing reference poses.
Walking has no reference pose for input zero, so its root-anchored external map
score remains withheld even under the available-reference cohort option.

## Held-out Desk 3 evidence

The frozen configurations processed 1,000 inputs from newly acquired Desk 3.
All 29 baseline losses are recovered by both alternatives, but fast introduces
two new losses and XFeat introduces seven. None of those recovered transitions
exceeds 10 cm / 5 degrees against reference relative motion.

| Policy | Losses | Emitted root-connected inputs | Shared-frame translation RMSE | Shared-frame rotation RMSE | Active loops outside 10 cm / 5 degrees |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLAHE | 29 | 550 | 22.22 mm | 1.638° | 0 / 57 |
| Fast | 2 | 774 | 24.77 mm | 2.215° | 0 / 47 |
| XFeat | 7 | 767 | 66.42 mm | 4.730° | 0 / 0 |

The shared population contains 550 connected inputs, of which 516 have reference
poses. Fast's whole emitted root has 50.67 mm translation RMSE and 4.513° rotation
RMSE. Its longer availability does not establish better global accuracy. XFeat
closes no loops on this prefix despite its improved local availability.

The original fixed 24-view map score is withheld because training references
170 and 190 are unavailable. An explicitly separate `--reference-cohort available`
comparison omits exactly those two views for every method, before fusion or
surface scoring. All three methods then use the same 22 training views. Capped
surface error is 43.10 mm for CLAHE, 41.73 mm for fast and 58.54 mm for XFeat.
The fraction explained within 5 cm is 79.81%, 80.80% and 72.40%. These are
native-depth/reference-camera diagnostics, not independent physical surface
truth. The strict cohort remains the evaluator default. All 10/6/6 saved chunks
for CLAHE/fast/XFeat pass exact refusion across all 100 admitted mapping frames.

This capture demonstrates transfer of the availability improvement, with mixed
accuracy and map effects. It does not justify a universal frontend promotion.

## Runtime, reproducibility and decision

Serial raw XYZ 160-input runs were repeated in opposite policy order, after the
concurrent matrix and its map evaluations finished. OpenCV, BLAS, OpenMP and the
learned CPU runtime use one thread. This machine has an Intel Core i7-13700HX.
The reported processing time excludes model construction, final export and
reference scoring. Peak RSS includes the loaded model/runtime.

| Policy | Elapsed time | Throughput | Processing p95 | Peak RSS |
| --- | --- | --- | --- | --- |
| Original depth kernel, CLAHE | 63.66–64.70 s | 2.47–2.51 inputs/s | 1.421–1.444 s | 216.4–217.7 MiB |
| Equivalent depth kernel, CLAHE | 41.80–41.91 s | 3.82–3.83 inputs/s | 1.143–1.153 s | 216.7–217.3 MiB |
| Fast | 19.50–19.52 s | 8.20–8.21 inputs/s | 0.558–0.561 s | 172.3–172.5 MiB |
| XFeat | 22.69–22.73 s | 7.04–7.05 inputs/s | 0.527–0.531 s | 376.4–388.2 MiB |

The equivalent depth implementation reduces this complete control run's time
by about 34–35%, with identical trajectory, loop and reconstruction arrays.
Every policy also reproduces its own arrays exactly across the two serial runs.
The faster optional frontends change scientific outcomes. Their timing advantage
cannot be interpreted as an accuracy-neutral replacement.

Longer runs retain growing history. At 1,000 Desk inputs, peak process RSS is
342 MiB for CLAHE, 300 MiB for fast and 634 MiB for XFeat. Place memory is at
its 64-entry limit, with 223/202/206 graph nodes. No policy reaches the 512-node
limit in this matrix. These memory observations do not establish bounded
long-duration operation. Full XYZ fast has a median processing latency near
60 ms but a 2.44 s p95 and a 5.74 s maximum in the concurrent matrix. Global
work remains a major serial latency source, even when typical tracking is fast.
The concurrent long-run times are not directly comparable to the serial probe.

The exact depth optimization is retained without changing its measurement
policy. Flow recovery and half-scale detection remain explicit options. XFeat
remains an experimental external adapter. Its CPU viability and availability
gains are demonstrated, while drift, memory cost and historical association
need further work. No GPU, learned matcher, online model training, live sensor
scheduler or outdoor generalization claim is made.

Verification includes the complete 1,002-test Linux suite and a subsequently
added odd-width resize regression, with 21 final targeted checks passing. Lint,
formatting and whitespace checks pass. Native Windows and hosted CI were not
run. All 21,267 protected pre-study scientific files retain their SHA-256 hashes.
The full XYZ default control reproduces the preceding baseline's trajectory,
loop constraints and reconstruction arrays exactly.

Fresh final-code XYZ 80-input prefixes for fast and XFeat, and ETH3D 120-input
prefixes for fast, exactly match the corresponding full runs' emitted poses,
components, local segments, source indices and versions. The original odd-width
prototype has different resampling, and its two cross-version differences are
retained explicitly in `resize_revision_audit.json`. Causal-prefix claims compare
compatible implementations. Source archives cover every raw-run package hash.
Final input/provenance guards preserve these valid-stream numerical results.

## Remaining architectural limits

The system remains serial. Global graph solves and map reconstruction can cause
latency spikes. Local descriptor aggregation is a weak historical retrieval
model, and neither geometric verification nor robust graph optimization excludes
all coherent false relationships. Calibration and depth bias remain correlated.
Local-pose summaries and loop groups do not form a calibrated independent noise
model. Global corrections remain placement transforms, without feeding graph
poses back as independent local observations.

Feature/model working state is bounded, but trajectory history, graph data up
to its explicit capacity, and reconstruction history remain separate memory
costs. The 512-node graph limit still stops new global queries while local
tracking continues. Persistent map storage, asynchronous bounded scheduling,
incremental factorization and stronger learned historical association remain
future work. Any promotion must account for accuracy, map quality and worst-case
latency alongside tracking availability.
