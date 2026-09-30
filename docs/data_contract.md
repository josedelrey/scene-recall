# RGB-D data contract

Dataset adapters are the boundary between dataset-specific formats and SceneRecall's canonical RGB-D data. This contract defines the output of that boundary for SceneRecall v1.

## Structure

| Entity | Scope | Required content |
| --- | --- | --- |
| `Sequence` | One ordered RGB-D capture | Opaque `sequence_id`, shared calibration, and observations |
| Shared calibration | One sequence | `K_R`, `K_D`, and `T_RD` |
| `Observation` | One associated RGB and depth pair | Canonical RGB view, canonical depth view, and `frame_index` |

An observation may also carry a timestamp, a reference pose, and provenance. `frame_index` is zero-based and contiguous within its sequence. The optional timestamp is a `float64` number of seconds and refers to the depth view. The adapter associates the RGB view with that depth view before producing the observation.

## Canonical views and frames

| View | Array contract | Optical frame |
| --- | --- | --- |
| RGB | `uint8`, RGB channel order, shape `(H_R, W_R, 3)` | $C_R$ |
| Depth | `float32` z-depth in meters, shape `(H_D, W_D)` | $C_D$ |

The canonical RGB view contains image values before model-specific normalization, resizing, or preprocessing. Canonical depth stores the camera-axis coordinate $Z=p_z^{C_D}$, never Euclidean range. Valid depth is finite and positive. Invalid depth is `NaN`.

$C_D$ is the reference frame of the observation: the optical frame in which its canonical depth view is parameterized. It need not be the physical depth or IR sensor frame. $C_R$ is the canonical RGB optical frame. Both use $+x$ right, $+y$ down, and $+z$ forward. RGB and depth need not share a resolution, pixel grid, intrinsics, or optical frame. In particular, aligning a depth view to the RGB grid can make $C_D=C_R$ even when the physical sensors differ.

## Shared calibration

SceneRecall uses $T_{AB}: B\rightarrow A$, with the destination frame first. The subscripts $R$ and $D$ in $T_{RD}$ denote $C_R$ and $C_D$.

| Field | Contract |
| --- | --- |
| $K_R$ | `float64`, shape `(3, 3)`, pinhole intrinsics for the canonical RGB pixel grid |
| $K_D$ | `float64`, shape `(3, 3)`, pinhole intrinsics for the canonical depth pixel grid |
| $T_{RD}$ | `float64`, homogeneous rigid transform of shape `(4, 4)`, translation in meters, mapping $C_D\rightarrow C_R$ |

Each intrinsic matrix is tied exactly to its corresponding canonical pixel grid and has zero skew in SceneRecall v1. Canonical views must be compatible with this pinhole model. An adapter that rectifies or changes a pixel grid must supply intrinsics for the resulting grid.

$T_{RD}$ is always conceptually defined and is the identity when the two canonical views share an optical frame. SceneRecall v1 assumes a rigid RGB-D rig. $K_R$, $K_D$, and $T_{RD}$ are constant within a sequence.

## Geometric relationships

For a valid depth pixel $(u_D,v_D)$ with z-depth $Z$, define

$$
\tilde{\mathbf{u}}_D=
\begin{bmatrix}
u_D \\
v_D \\
1
\end{bmatrix},
\qquad
\mathbf{p}^{C_D}=ZK_D^{-1}\tilde{\mathbf{u}}_D.
$$

The same point expressed in the RGB optical frame is

$$
\mathbf{p}^{C_R}=T_{RD}\mathbf{p}^{C_D}.
$$

If a reference pose is available, the world-frame point is

$$
\mathbf{p}^{W}=T_{WC_D}\mathbf{p}^{C_D}.
$$

Applying a homogeneous transform to a 3D point denotes its affine action, as in the camera geometry convention.

## Optional reference pose and provenance

$T_{WC_D}: C_D\rightarrow W$ is an optional reference pose for an observation. It is a `float64` homogeneous rigid transform of shape `(4, 4)` with translation in meters. An observation is valid without a known global pose. Reference poses remain distinct from poses estimated by SceneRecall odometry.

Provenance is optional and non-normative. It may record source IDs, filenames, original timestamps, or other dataset metadata. Core algorithms must never depend on provenance to interpret canonical data.

## Adapter responsibilities

Adapters parse source data, associate RGB and depth in time, and normalize depth units and conventions, invalid values, pose direction and representation, RGB channel order, and calibration into this contract. Source range measurements must be converted to z-depth. Adapters must account for source-specific distortion, rectification, or pre-registration as needed so that the supplied canonical views and intrinsics are geometrically consistent. The contract does not require a universal rectification or registration strategy.

Adapters do not perform downstream or model-specific preprocessing. All internal geometric quantities use meters. SceneRecall core code must not contain dataset-specific parsing, naming, synchronization, unit-conversion, calibration, or pose-convention logic.
