# RECONSTRUCTION_SPEC.md

## 0. Mission

Build an **automatic real-to-simulation reconstruction pipeline** inspired by DexGPT:

> paired MP4/GIF recordings → synchronized, camera-compensated 3D interaction reconstruction → one fused MuJoCo simulation

Primary reference: https://github.com/Hu-xiao-max/dexgpt

Use DexGPT's architectural ideas—observation extraction, temporal smoothing, retargeting/IK, MuJoCo contact rollout, inspectable outputs, and quantitative validation—but generalize from **two robot hands + one hinged object** to:

- **Guider:** head pose, rigid torso/chest pose, both arms, both articulated hands/fingers
- **Builder:** visible forearms, both articulated hands/fingers
- **Environment:** table + all blocks
- **Interaction:** blocks must move through MuJoCo hand contact, not scripted block pose playback

Do not copy DexGPT code/assets unless their licensing explicitly permits it. Reimplement the pipeline cleanly.

---

## 1. Non-negotiable requirements

1. Inputs: `.mp4` and `.gif`.
2. Primary mode accepts a **paired guider video + builder video of the same interaction** and fuses both into one scene.
3. The input cameras are moving Meta Aria glasses; assume only exported video/audio is available unless extra calibration files appear.
4. The pipeline must be **fully automatic**:
   - no manual calibration,
   - no frame-by-frame annotation,
   - no hand-picked table corners,
   - no manually entered block sizes,
   - no manually entered synchronization offset.
5. Output must be a MuJoCo reconstruction.
6. Guider simulation:
   - head position/orientation,
   - rigid torso position/orientation,
   - shoulders/arms,
   - wrists,
   - individual finger articulation.
   - No facial-expression or mouth reconstruction.
7. Builder simulation:
   - visible forearms,
   - wrists,
   - articulated hands/fingers.
   - Do not model the builder's full body unless needed internally for tracking.
8. Blocks must be free dynamic bodies after initialization. **Never overwrite block pose during the physics rollout.**
9. Builder hand motion must cause observed block motion through collision/contact/friction.
10. Primary objective is **visual + physical reconstruction**, not gesture semantics. Do not add intent labels such as "point", "move left", etc. unless useful internally.
11. Keep the implementation modular and efficient. Prefer a reliable baseline over a large research stack.

---

## 2. Test archive facts

`test_1.zip` currently contains **6 usable MP4s**, not 5:

```text
segment_002_instruction_candidate.mp4
segment_003_instruction_candidate.mp4
segment_003_assembler_ego_with_audio.mp4
segment_012_assembler_ego_with_audio.mp4
segment_021_assembler_ego_with_audio.mp4
segment_027_assembler_ego_with_audio.mp4
```

Metadata observed in the archive:

| File | Resolution | FPS | Duration |
|---|---:|---:|---:|
| segment_002_instruction_candidate.mp4 | 704×704 | 20 | 16.30 s |
| segment_003_instruction_candidate.mp4 | 704×704 | 20 | 12.45 s |
| segment_003_assembler_ego_with_audio.mp4 | 768×768 | 15 | 5.40 s |
| segment_012_assembler_ego_with_audio.mp4 | 768×768 | 15 | 14.00 s |
| segment_021_assembler_ego_with_audio.mp4 | 768×768 | 15 | 8.67 s |
| segment_027_assembler_ego_with_audio.mp4 | 768×768 | 15 | 5.93 s |

All six contain audio.

Only `segment_003` has both naming roles in this archive, so it is the first required fused benchmark. The program must still discover all files and report unmatched segments without crashing.

Do **not** hard-code segment 003.

---

## 3. Required user-facing command

Target:

```bash
python -m interaction_recon run \
  --input test_1.zip \
  --output outputs/test_1
```

Also support explicit pairs:

```bash
python -m interaction_recon reconstruct \
  --guider path/to/segment_003_instruction_candidate.mp4 \
  --builder path/to/segment_003_assembler_ego_with_audio.mp4 \
  --output outputs/segment_003
```

GIFs must work through the same interface.

Optional:

```bash
--render-format mp4|gif|both
--device auto|cuda|cpu
--debug
```

No manual geometry/calibration arguments should be required for the normal pipeline.

---

## 4. Automatic file discovery and pairing

### 4.1 Archive handling

If input is a ZIP:

- extract to a temporary working directory,
- ignore `__MACOSX`, `.DS_Store`, AppleDouble files, and hidden junk,
- recursively discover `.mp4`, `.gif`.

### 4.2 Pairing

Parse a segment identifier with a regex equivalent to:

```text
segment_(\d+)
```

Role inference:

```text
*instruction_candidate*      -> guider stream
*assembler_ego_with_audio*   -> builder stream
```

Group by segment ID.

For each group:

```json
{
  "segment_id": "003",
  "guider": "...instruction_candidate.mp4",
  "builder": "...assembler_ego_with_audio.mp4",
  "status": "paired"
}
```

If one stream is missing:

```json
{
  "segment_id": "012",
  "status": "unpaired"
}
```

Unpaired files must be written to the manifest and may optionally receive a single-view diagnostic reconstruction, but **must not be presented as a fused two-person result**.

---

## 5. High-level pipeline

```text
MP4/GIF pair
    |
    v
decode frames + audio + timestamps
    |
    +--> automatic temporal synchronization
    |
    +--> per-camera pose/intrinsics estimation
    |
    +--> guider/body/hand tracking
    |
    +--> builder forearm/hand tracking
    |
    +--> table + block detection/tracking
    |
    v
common world-frame reconstruction
    |
    v
temporal filtering + missing-data handling
    |
    v
human-to-MuJoCo retargeting / IK
    |
    v
kinematic reference reconstruction
    |
    v
contact-driven MuJoCo rollout
    |
    v
automatic physics refinement
    |
    v
render + metrics + artifacts
```

Keep **observations**, **retargeting**, and **physics rollout** as separate artifacts, as DexGPT does.

---

## 6. Stage A — media normalization

Implement one media loader.

For each source:

```python
MediaSequence:
    rgb_frames
    timestamps_s
    fps
    width
    height
    audio_waveform
    audio_sample_rate
```

Requirements:

- preserve original timestamps,
- decode variable FPS correctly if encountered,
- standardize tracking to a configurable analysis FPS, e.g. 15–20 FPS,
- preserve original duration for final render,
- GIFs receive synthetic timestamps from frame durations,
- MP4 audio is extracted when available.

Do not make later modules depend on whether the input was MP4 or GIF.

---

## 7. Stage B — automatic synchronization

The two videos may start/end at different times.

### Primary method: audio

Use normalized audio cross-correlation / GCC-PHAT to estimate temporal offset.

Steps:

1. resample both audio tracks to one mono sample rate,
2. remove DC / normalize,
3. compute coarse offset,
4. refine around the best lag,
5. produce `sync_offset_s` and confidence.

### Fallback: visual motion

If audio is absent or confidence is low:

- compute hand/block motion-energy time series,
- align with normalized cross-correlation / dynamic time warping,
- prefer block-manipulation events.

Output:

```json
{
  "offset_s": ...,
  "method": "audio_gcc_phat|visual_motion",
  "confidence": ...
}
```

Never assume equal start time.

---

## 8. Stage C — moving-camera reconstruction

Because both streams come from head-mounted Aria cameras, **camera motion must be separated from scene motion**.

### Goal

For every frame estimate:

```text
K_t : camera intrinsics (usually shared/slowly varying)
T_world_from_camera(t) : 6-DoF camera pose
```

### Preferred baseline

Use a visual-SLAM / SfM backend capable of wide-FOV imagery, e.g.:

- PyCOLMAP/COLMAP with a radial/fisheye camera model, or
- another local open-source SLAM backend if more robust.

Do not depend on Aria VRS metadata because the required input is MP4/GIF.

### Dynamic-scene handling

Exclude or down-weight:

- hands,
- arms,
- torso/person,
- moving blocks

from camera-pose feature matching whenever masks are available.

Use static features from:

- table texture,
- room,
- monitor/chairs/walls,
- other background.

If SLAM fails on a short clip:

1. estimate inter-frame homography/essential motion on static-background features,
2. stabilize relative camera motion,
3. continue with lower-confidence output rather than silently fabricating a pose.

Record camera tracking confidence per frame.

---

## 9. Stage D — human tracking

### 9.1 Guider

Required observed targets:

```text
head center + orientation
left/right shoulder
left/right elbow
left/right wrist
torso center + orientation
left/right hand: 21 landmarks each
```

A lightweight first baseline may use MediaPipe Pose/Holistic + Hand Landmarker.

For better 3D accuracy, keep the detector interface swappable so a stronger body/hand model can replace it without changing downstream code.

### 9.2 Builder

Required:

```text
left/right elbow when visible
left/right wrist
left/right hand: 21 landmarks each
```

Only render forearm + hand.

### 9.3 Head

Do not reconstruct expressions.

Estimate only:

```text
position
rotation quaternion
```

Use stable face/head landmarks or head-pose estimation.

### 9.4 Torso

Represent guider torso as one rigid body.

Estimate:

```text
position
rotation quaternion
scale
```

Orientation should be derived from shoulder axis + vertical body axis, not facial expression.

### 9.5 Hands

Each hand must retain individual finger articulation.

Minimum observations:

```text
21 keypoints x (u, v, relative_depth, confidence)
```

Do not collapse the hand to one rigid palm.

---

## 10. Stage E — table and block reconstruction

### 10.1 Table

Automatically estimate:

- visible table mask/plane,
- plane normal,
- table top coordinate frame,
- approximate extents.

Define the canonical world frame:

```text
z = up from table
z = 0 at table surface
x/y = axes in the table plane
```

Use table edges when reliable; otherwise use PCA/scene alignment.

### 10.2 Blocks

Automatically detect all block-like objects on the table.

Use a modular strategy:

1. segmentation,
2. filter masks using table region + compact rectangular geometry,
3. track identities across frames,
4. fit each object with an oriented rectangular cuboid.

A practical baseline may combine:

- OpenCV geometry/color cues for these simple light blocks,
- segmentation/video-mask tracking as a fallback.

For each block and synchronized timestamp estimate:

```text
block_id
center_xyz
rotation_wxyz
dimensions_xyz
visibility
confidence
```

### 10.3 Unknown scale

No exact table/block dimensions are given.

Estimate scene scale automatically using multiple weak priors, then optimize consistency:

- adult hand bone-length prior,
- guider upper-arm/forearm anthropometric prior,
- block constancy across frames/views,
- table-plane consistency,
- multi-view correspondence.

Store scale uncertainty.

Do **not** claim millimeter-level metric accuracy from monocular video.

---

## 11. Stage F — fuse both videos into one world

After synchronization and per-camera reconstruction:

1. transform observations into each camera's reconstructed world,
2. align table planes,
3. use synchronized shared geometry—blocks, table features, and any jointly visible body points—to estimate a robust similarity transform,
4. use RANSAC + Umeyama/Procrustes alignment,
5. jointly refine:
   - camera alignment,
   - common scene scale,
   - block trajectories,
   - shared visible keypoints.

Result:

```text
one canonical table coordinate system
one guider
one builder
one set of blocks
two moving source cameras
```

When the same joint/object is visible in both cameras at the same synchronized time, triangulate/fuse it.

When only one camera sees it, fall back to:

- monocular depth,
- kinematic bone-length constraints,
- table/contact constraints,
- temporal continuity.

---

## 12. Stage G — filtering and missing data

Raw vision estimates will jitter.

For every tracked quantity preserve:

```text
value
confidence
observed_mask
interpolated_mask
source_camera(s)
```

Pipeline:

```text
raw detection
-> outlier rejection
-> short-gap interpolation
-> temporal smoothing
-> kinematic/scene constraints
```

Rules:

- smooth in 3D, not only image space,
- preserve fast intentional finger motion,
- do not interpolate long occlusions as if observed,
- long missing intervals must carry low confidence,
- quaternions use rotation-aware interpolation.

---

## 13. Stage H — MuJoCo human models

Do not chase photorealism. Use simple collision-safe geometry.

### 13.1 Guider model

```text
torso: rigid box/capsule
head: rigid sphere/ellipsoid
upper arms: capsules
forearms: capsules
hands: articulated palm + phalanges
```

Suggested DOFs:

- torso root: 6 DoF target pose
- head/neck: 3 DoF
- each shoulder: 3 DoF
- each elbow: 1–2 DoF
- each wrist: 2–3 DoF
- each hand: articulated thumb + four fingers, ~20+ finger DoF

### 13.2 Builder model

```text
left forearm -> wrist -> articulated hand
right forearm -> wrist -> articulated hand
```

No builder torso/head in rendering.

Builder wrist roots may be free dynamic bodies driven toward observed targets through compliant constraints/controllers.

### 13.3 Hand collisions

Use stable primitive/convex collision bodies on:

- palm,
- each phalanx,
- fingertips/finger pads.

Visual meshes are optional. Collision geometry matters more.

---

## 14. Stage I — retargeting and IK

Convert observed 3D keypoints into simulator joint targets.

For each frame solve approximately:

```text
L =
  landmark_position_error
+ fingertip_error
+ wrist_orientation_error
+ joint_limit_penalty
+ temporal_smoothness
+ self_collision_penalty
```

For arms:

- match shoulders/elbows/wrists.

For hands:

- match wrist/palm orientation,
- preserve finger bone directions,
- fit fingertip and intermediate-joint targets,
- obey anatomical joint limits.

Save the result before physics:

```text
retargeted.npz
```

The kinematic reference is allowed to prescribe the observed block trajectory **only for visualization/evaluation**, never for the final contact rollout.

---

## 15. Stage J — MuJoCo physics reconstruction

This is mandatory.

### 15.1 Blocks

Each block:

- free joint,
- collision enabled,
- gravity enabled,
- plausible density/mass,
- friction enabled,
- no actuator,
- no weld to a hand,
- no pose overwrite after initial state.

### 15.2 Hands

Follow the DexGPT philosophy:

- hands/wrists are dynamic,
- wrists are driven toward reference trajectories with compliant controls/constraints,
- finger joints use position targets with finite actuator limits,
- collision/contact/friction remain active.

The simulator should be able to deviate from reference if contact dynamics require it.

### 15.3 Physics timestep

Use a physics timestep substantially smaller than video sampling, e.g.:

```text
0.001–0.002 s physics
15–20 Hz observation/reference
30–60 FPS render
```

Interpolate control targets between observation frames.

---

## 16. Stage K — automatic inverse-physics refinement

A single forward rollout will probably not reproduce block motion accurately. Automatically refine the simulation.

### Parameters allowed to optimize

Within conservative bounded ranges:

```text
block mass/density
block-table friction
block-hand friction
contact softness/solref/solimp
wrist controller stiffness/damping
finger controller gains
small hand pose offsets
small grasp-timing offsets
```

Do not optimize by directly controlling block pose.

### Objective

Compare simulated free blocks to the reconstructed visual-reference trajectory:

```text
L_physics =
    w1 * block_position_error
  + w2 * block_rotation_error
  + w3 * contact_timing_error
  + w4 * hand_reference_deviation
  + w5 * penetration_penalty
  + w6 * joint_limit_penalty
  + w7 * instability_penalty
```

Use a derivative-free optimizer suitable for a small parameter space, e.g. CMA-ES / Optuna / scipy differential evolution.

Use coarse-to-fine refinement:

1. low-FPS/short rollout,
2. best candidates,
3. full-resolution rollout.

Cache vision results so physics search never reruns expensive perception.

---

## 17. Rendering

Primary required artifact:

```text
simulation.mp4
```

It must show **both actors together in one canonical MuJoCo scene**:

- guider across table,
- builder forearms/hands near table,
- blocks,
- table.

Also produce:

```text
comparison.mp4
```

Recommended layout:

```text
source guider | source builder | fused MuJoCo reconstruction
```

Optional debug render:

```text
tracking.mp4
```

with source-frame overlays for:

- pose,
- hands,
- block masks/IDs,
- table plane,
- camera confidence.

If `--render-format gif` or `both`, convert the final simulation to GIF after MP4 rendering.

---

## 18. Required outputs

Per paired segment:

```text
outputs/<segment_id>/
├── manifest.json
├── sync.json
├── cameras.npz
├── observations.npz
├── retargeted.npz
├── scene.xml
├── physics_rollout.npz
├── metrics.json
├── simulation.mp4
├── comparison.mp4
├── tracking.mp4
└── logs/
```

### `observations.npz`

At minimum:

```text
timestamps
guider_head_pose
guider_torso_pose
guider_arm_keypoints
guider_left_hand_21
guider_right_hand_21
builder_forearm_keypoints
builder_left_hand_21
builder_right_hand_21
block_poses
block_dimensions
table_plane
confidence arrays
observed masks
interpolated masks
camera/source provenance
```

### `retargeted.npz`

At minimum:

```text
timestamps
guider_qpos_targets
builder_qpos_targets
wrist/root poses
joint names
IK residuals
```

### `physics_rollout.npz`

At minimum:

```text
timestamps
qpos
qvel
ctrl
block poses
contacts
contact forces
penetration/depth diagnostics
```

---

## 19. Metrics

Always report failures quantitatively. Never hide bad reconstruction behind a plausible render.

### Perception

```text
hand detection coverage
body-keypoint coverage
block tracking coverage
camera-pose coverage
synchronization confidence
2D reprojection error
```

### Reconstruction

```text
cross-view landmark consistency
table-plane consistency
block dimension variance
IK residual
joint-limit violations
```

### Physics

```text
block trajectory position error
block rotation error
error normalized by block length
contact-frame fraction
max penetration
peak contact force
joint-limit overshoot
simulation stability / NaNs
```

Because absolute scale is estimated, always include both:

- metric error using estimated scale,
- scale-normalized error.

---

## 20. Development targets

These are engineering targets, not claims of ground-truth physical accuracy.

For the paired `segment_003` benchmark:

1. Pipeline runs end-to-end with **zero manual input** beyond the ZIP/path.
2. Correctly discovers 6 videos and identifies segment 003 as paired.
3. Produces a single fused MuJoCo scene containing:
   - guider head,
   - guider torso,
   - guider arms,
   - both guider hands/fingers,
   - builder forearms,
   - both builder hands/fingers,
   - table,
   - all detected blocks.
4. Final physics rollout is stable: no NaNs/exploding bodies.
5. Blocks are never actuated, welded to hands, or pose-overwritten after initialization.
6. Block motion in final rollout comes from hand contact.
7. Tracking/reconstruction confidence and all metrics are written even when poor.
8. `simulation.mp4` and `comparison.mp4` are generated automatically.
9. Automated tests confirm the no-block-actuation rule.
10. Missing/unpaired segments are reported cleanly instead of causing failure.

Useful target thresholds once the baseline works:

```text
visible-hand tracking coverage: >= 80%
mean visible landmark reprojection: <= 15 px
block centroid reprojection: <= 20 px
physics block trajectory error: <= 0.25 block lengths initially
max persistent penetration: <= 10% of smallest block dimension
```

If a target is missed, keep the run and report the measured value.

---

## 21. Suggested repository layout

```text
interaction-recon/
├── README.md
├── RECONSTRUCTION_SPEC.md
├── pyproject.toml
├── requirements.txt
├── interaction_recon/
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── io/
│   │   ├── discover.py
│   │   ├── media.py
│   │   └── sync.py
│   ├── vision/
│   │   ├── camera.py
│   │   ├── body.py
│   │   ├── hands.py
│   │   ├── table.py
│   │   └── blocks.py
│   ├── fusion/
│   │   ├── world.py
│   │   ├── triangulation.py
│   │   └── filtering.py
│   ├── retarget/
│   │   ├── arm_ik.py
│   │   ├── hand_ik.py
│   │   └── models.py
│   ├── simulation/
│   │   ├── build_scene.py
│   │   ├── controllers.py
│   │   ├── rollout.py
│   │   └── refine_physics.py
│   ├── render/
│   │   ├── render.py
│   │   └── comparison.py
│   └── eval/
│       ├── metrics.py
│       └── audit.py
├── assets/
│   └── mujoco/
├── tests/
│   ├── test_discovery.py
│   ├── test_sync.py
│   ├── test_retarget.py
│   ├── test_scene.py
│   └── test_no_block_actuation.py
└── outputs/
```

---

## 22. Dependency strategy

Keep core dependencies minimal.

Likely baseline:

```text
python 3.11
numpy
scipy
opencv-python
mujoco
mediapipe
torch
trimesh
imageio / imageio-ffmpeg
soundfile
librosa or scipy audio utilities
pycolmap (camera reconstruction)
```

Optional heavier vision models must sit behind interfaces and must not be required unless they clearly improve the benchmark.

Prefer local/open-source inference. Do not add an LLM/API dependency to the reconstruction loop.

---

## 23. Implementation order for Astra

Implement in this order. Do not begin inverse-physics optimization before the upstream geometry is inspectable.

### Milestone 1 — scaffold + data contract

- CLI
- ZIP/media discovery
- pairing
- metadata extraction
- output directories
- tests

Pass when:

```bash
python -m interaction_recon run --input test_1.zip --output outputs/test_1
```

discovers all six files and emits a correct manifest.

### Milestone 2 — synchronization + tracking overlays

- audio sync
- body/hands
- blocks/table
- `tracking.mp4`

Pass when segment 003 has synchronized streams and inspectable tracking overlays.

### Milestone 3 — moving-camera + common 3D world

- camera motion
- table frame
- scale estimate
- cross-view alignment
- 3D observations

Pass when both streams render coherently into one canonical table frame.

### Milestone 4 — MuJoCo kinematic reference

- guider skeleton
- builder forearms/hands
- articulated fingers
- blocks/table
- IK/retargeting
- kinematic reference render

Pass when the reconstructed motion visually follows the videos.

### Milestone 5 — contact physics

- free blocks
- dynamic/contact-capable builder hands
- controller interpolation
- contact rollout
- no block pose overwrite

Pass when blocks move only through contact and simulation stays stable.

### Milestone 6 — automatic physics refinement

- optimize bounded physical/controller parameters
- compare simulated block motion to visual reference
- select best valid rollout

Pass when refinement measurably improves block trajectory error over the initial rollout.

### Milestone 7 — final render + audit

- `simulation.mp4`
- `comparison.mp4`
- metrics
- automated audit
- README quickstart

---

## 24. Critical engineering rules

1. **Never mix observation truth with simulation state.**
   - `observations.npz` = what vision estimated.
   - `retargeted.npz` = target human motion.
   - `physics_rollout.npz` = what MuJoCo actually did.

2. **Never fake physics success.**
   The kinematic reference may replay block motion for comparison only. The final physics rollout may not.

3. **Cache expensive stages.**
   Rerunning physics must not rerun video tracking or SLAM.

4. **Every inferred quantity gets confidence/provenance.**
   Especially scale, depth, camera pose, interpolated joints, and occluded blocks.

5. **Make debug videos early.**
   A correct visual overlay is more useful than optimizing invisible numeric arrays.

6. **Use simple simulator geometry first.**
   Primitive capsules/boxes are preferred over photorealistic meshes.

7. **Do not hard-code the current room, actor appearance, timestamps, or segment number.**

8. **Fail loudly but usefully.**
   If SLAM, synchronization, or hand tracking is weak, emit diagnostics and confidence instead of silently using arbitrary constants.

---

## 25. Definition of done for V1

V1 is complete when a clean checkout can run:

```bash
python -m interaction_recon run \
  --input test_1.zip \
  --output outputs/test_1
```

with no manual calibration and automatically:

1. finds and pairs compatible videos,
2. synchronizes the paired views,
3. estimates moving camera trajectories,
4. reconstructs one shared table/world frame,
5. estimates table and block geometry,
6. reconstructs guider head/torso/arms/articulated hands,
7. reconstructs builder forearms/articulated hands,
8. retargets both actors to MuJoCo,
9. initializes free block bodies,
10. drives hand motion,
11. produces block motion through contact physics,
12. automatically refines bounded physical/controller parameters,
13. renders both actors together,
14. saves `simulation.mp4`, `comparison.mp4`, structured states, and quantitative metrics,
15. reports unpaired clips without crashing.

The first required fused benchmark is `segment_003`.

---

## 26. Out of scope for V1

Do not spend time on:

- facial expressions,
- lip motion,
- photorealistic avatars,
- speech/gesture semantics,
- instruction understanding,
- full builder body,
- robot hardware control,
- generative video synthesis,
- perfect metric-scale recovery,
- arbitrary household-object reconstruction,
- replacing MuJoCo with Blender.

The V1 goal is narrow:

> **Automatically reconstruct the observed two-person block-building interaction as a visually faithful, contact-driven MuJoCo simulation.**
