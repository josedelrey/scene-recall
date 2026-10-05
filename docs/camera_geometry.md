# Camera geometry

This document defines the coordinate, projection, depth, and rigid-transform conventions used throughout SceneRecall for geometry, odometry, mapping, and RGB-D processing.

## Coordinate frames

$C$ is the camera optical frame. Its $+x$ axis points right in the image, $+y$ points down, and $+z$ points forward. $W$ is the persistent world frame. A point expressed in these frames is denoted $\mathbf{p}^C$ or $\mathbf{p}^W$, respectively.

## Transforms

The first subscript of a transform is its destination frame. The second is its source frame:

$$
T_{AB}: B \rightarrow A,
\qquad
\mathbf{p}^A = T_{AB}\mathbf{p}^B.
$$

Thus, $T_{WC}$ maps camera coordinates to world coordinates, and $T_{CW}$ maps world coordinates to camera coordinates. Applying $T$ to a 3D point denotes its affine action. Matrix multiplication by the homogeneous $4\times4$ representation uses the point's homogeneous lift.

For a rigid camera pose,

$$
\mathbf{p}^W = R_{WC}\mathbf{p}^C + \mathbf{t}_{WC},
\qquad
T_{WC} =
\begin{bmatrix}
R_{WC} & \mathbf{t}_{WC} \\
\mathbf{0}^T & 1
\end{bmatrix}.
$$

$R_{WC}$ expresses the orientation of $C$ in $W$. $\mathbf{t}_{WC}$ is the position of the origin of $C$ expressed in $W$. Here $R\in SO(3)$ and $T\in SE(3)$, with

$$
R^T R = I,
\qquad
\det(R)=1,
\qquad
R^{-1}=R^T.
$$

The inverse pose is

$$
T_{CW}=T_{WC}^{-1},
\qquad
R_{CW}=R_{WC}^T,
\qquad
\mathbf{t}_{CW}=-R_{WC}^T\mathbf{t}_{WC}.
$$

The inverse translation is generally not $-\mathbf{t}_{WC}$ because it must be expressed in $C$.

The rotation identities and transpose-based inverse above assume a rigid pose. The [canonical data contract](data_contract.md#transform-validation-policy) checks homogeneous structure without enforcing a numerical $SO(3)$ tolerance or repairing rotations. Consumers that rely on these identities must establish the rotation quality they require.

Transforms compose in source-to-destination order:

$$
T_{AB}T_{BC}=T_{AC},
\qquad
T_{WC}=T_{WB}T_{BC}.
$$

Transform order matters. In the second equation, $B$ is an intermediate frame.

## Pinhole projection and intrinsics

For a point in front of the camera,

$$
\mathbf{p}^C=
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix},
\qquad Z>0,
\qquad
x=\frac{X}{Z},
\qquad
y=\frac{Y}{Z}.
$$

The normalized image plane is at $z=1$. A normalized image point identifies a 3D viewing ray. Projection removes depth.

For an ideal pinhole camera without skew,

$$
K=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix},
\qquad
u=f_x\frac{X}{Z}+c_x,
\qquad
v=f_y\frac{Y}{Z}+c_y.
$$

$f_x$ and $f_y$ are focal lengths in pixels. $(c_x,c_y)$ is the principal point in pixel coordinates. In homogeneous image notation,

$$
\tilde{\mathbf{u}}=
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix},
\qquad
\tilde{\mathbf{u}}\sim K\mathbf{p}^C.
$$

The symbol $\sim$ means equality up to a nonzero scale.

## RGB-D backprojection and depth convention

For pixel $(u,v)$ and **z-depth** $Z$,

$$
K^{-1}\tilde{\mathbf{u}}=
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix},
\qquad
\mathbf{p}^C=ZK^{-1}\tilde{\mathbf{u}}.
$$

Equivalently,

$$
X=Z\frac{u-c_x}{f_x},
\qquad
Y=Z\frac{v-c_y}{f_y}.
$$

Z-depth is the camera-axis coordinate $Z=p_z^C$. Range is the Euclidean distance $r=\|\mathbf{p}^C\|_2$. The backprojection above assumes z-depth. For range input, use

$$
\mathbf{d}=K^{-1}\tilde{\mathbf{u}},
\qquad
\mathbf{p}^C=r\frac{\mathbf{d}}{\|\mathbf{d}\|_2}.
$$

## Camera and world geometry

The RGB-D mapping chain is

$$
(u,v,Z)\rightarrow\mathbf{p}^C\rightarrow\mathbf{p}^W,
\qquad
\mathbf{p}^W=R_{WC}\mathbf{p}^C+\mathbf{t}_{WC}.
$$

This places per-frame RGB-D geometry into a persistent world-frame map.

To project a world point into the camera, first compute $\mathbf{p}^C=T_{CW}\mathbf{p}^W$, then apply pinhole projection. Let $\tilde{\mathbf{p}}^W=[(\mathbf{p}^W)^T,1]^T$. The compact homogeneous form is

$$
\tilde{\mathbf{u}}\sim K T_{CW}\tilde{\mathbf{p}}^W.
$$

In this projection expression, $K T_{CW}\tilde{\mathbf{p}}^W$ abbreviates $K[I_3\;\mathbf{0}]T_{CW}\tilde{\mathbf{p}}^W$, which takes the first three coordinates of the transformed homogeneous point before applying $K$. Projection requires positive camera-frame $Z$.

## Depth and calibration assumptions

Each dataset or sensor adapter must explicitly define:

- Depth convention: z-depth or range
- Depth units and scale
- Invalid-depth representation
- RGB and depth alignment
- Image resolution associated with the intrinsics
- Distortion and rectification assumptions
- Pose convention: $T_{WC}$ or $T_{CW}$

SceneRecall's internal geometric calculations use meters. External dataset variable names do not establish their transform convention.
