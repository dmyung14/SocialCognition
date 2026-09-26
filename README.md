# Interaction reconstruction

Automatic MP4/GIF discovery, synchronization, uncertain 3D reconstruction,
articulated-human retargeting, contact-driven MuJoCo simulation, bounded physics
refinement, rendering and integrity auditing.

This is an inspectable engineering baseline, **not a calibrated motion-capture
system or a verified reconstruction of the supplied archive's shared interaction**.
A plausible movie is not evidence of accurate reconstruction.

## Installation

The current package requires Python 3.14 or later. The integration environment
uses Windows 11, Python 3.14.5, MuJoCo 3.13, OpenCV-contrib, NumPy, SciPy,
MediaPipe Tasks and imageio-ffmpeg.

No system ffmpeg/ffprobe, PyCOLMAP, Torch, librosa or soundfile is required.
FFmpeg comes from `imageio_ffmpeg.get_ffmpeg_exe()`.
An OpenGL-capable graphics driver is required for MuJoCo rendering.

### Windows / PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
New-Item -ItemType Directory -Force assets/models
```

If activation is unavailable, invoke `.\.venv\Scripts\python.exe` directly.

### bash

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
mkdir -p assets/models
```

On a headless Linux installation, configure a supported MuJoCo OpenGL backend
such as EGL before running. Windows desktop rendering uses the installed OpenGL
driver; the pipeline does not require CUDA.

## Required MediaPipe models

Place these official MediaPipe Tasks model bundles in `assets/models/`:

| Local filename | Official download |
|---|---|
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| `pose_landmarker_full.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task |
| `face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |

The model cards and usage documentation are at
https://ai.google.dev/edge/mediapipe/solutions/vision .
Review their terms before redistribution. Models are gitignored, not bundled
with this repository, and are not downloaded implicitly.

PowerShell example:

```powershell
$base = "https://storage.googleapis.com/mediapipe-models"
Invoke-WebRequest "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile assets/models/hand_landmarker.task
Invoke-WebRequest "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile assets/models/pose_landmarker_full.task
Invoke-WebRequest "$base/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile assets/models/face_landmarker.task
```

bash equivalent:

```bash
base=https://storage.googleapis.com/mediapipe-models
curl -fL "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o assets/models/hand_landmarker.task
curl -fL "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o assets/models/pose_landmarker_full.task
curl -fL "$base/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -o assets/models/face_landmarker.task
```

Use `--model-dir PATH` for a different directory. Missing models produce an
explicit error naming the missing files. Only the MediaPipe **Tasks API** is used;
`mp.solutions` is not required.

## Quickstart

The following single-line commands work in both PowerShell and bash:

```text
python -m interaction_recon run --input test_1.zip --output outputs/test_1
python -m interaction_recon run --input test_1/test_1 --output outputs/test_1
python -m interaction_recon reconstruct --guider path/to/guider.mp4 --builder path/to/builder.mp4 --output outputs/pair
```

GIF inputs use the same interface. Explicit pairs do not require role filenames.

Useful options:

```text
--render-format mp4|gif|both
--analysis-fps 15
--model-dir assets/models
--force
--debug
--inventory-only
--device auto|cpu|cuda
```

`gif` and `both` **retain MP4 outputs** and additionally convert simulation and
comparison movies to palette-based GIFs, at no more than 15 FPS and 640 pixels
wide. `--device` is accepted for compatibility; detection, reconstruction, IK
and physics currently run on CPU. Rendering uses OpenGL.

Inventory-only mode needs no models. ZIP extraction is temporary; manifests
retain persistent archive/member locators rather than temporary paths.

The root manifest includes discovery records, unpaired/ambiguous segments and
a `segment_status_table`. Unpaired segments are skipped, not presented as fused
results, and do not require model loading.

## Pipeline stages

1. **Media:** decode actual presentation timestamps and audio using bundled
   FFmpeg. GIF timing comes from frame delays.
2. **Synchronization:** audio GCC-PHAT with waveform-coherence confidence;
   visual motion fallback. The convention is
   `builder_source_time = guider_source_time + offset_s`.
3. **Perception:** Tasks body, hand and head landmarks; OpenCV table and block
   segmentation/tracking. Occluded holds remain distinct from observations.
4. **Camera/world:** masked static-background OpenCV essential/homography
   motion; uncertain intrinsics, anthropometric metric-depth and table priors.
   Cross-view identity is not assumed merely because filenames match.
5. **Retargeting:** articulated guider and builder forearms/hands; bounded IK,
   per-chain residuals, short holds and explicitly flagged neutral rest.
6. **Physics:** initialize free blocks once, settle, then advance with MuJoCo.
   Dynamic builder hands use finite-force joint controllers and compliant
   root welds to hand targets. Blocks have no actuators, welds or mocap control.
   The visual-only guider does not collide with blocks.
7. **Refinement:** bounded serial CMA-ES over physical/controller parameters
   and small hand-reference offsets. Short candidates simulate the original
   prefix; no candidate resets blocks to a later observed pose.
8. **Final render/audit:** auto-framed 30 FPS movies, synchronized comparison,
   optional GIFs, structured metrics and automated integrity checks.

The overview camera uses a builder-side, elevated three-quarter view and bounds
from the entire recorded motion. It aims for 70% table width, widening when
necessary to retain actor bounds. Default simulation/reference movies are
960×720. Lighting and color changes are render-only and cannot alter physics.

The comparison shows equal-height panels:

```text
source guider | source builder | fused MuJoCo reconstruction
```

Source panels use original PTS and display `no source frame` outside their own
ranges. Its status bar reports synchronization method/confidence, fusion method
and physics error. `PHYSICS (refined)` is used only for an actually selected valid
refinement; otherwise the initial-retained status is shown.

## Outputs

For an automatically discovered pair, files are under
`outputs/test_1/<segment_id>/`. An explicit pair writes directly to its requested
output directory.

| Artifact | Meaning |
|---|---|
| `manifest.json` | Stage status, sources, warnings, artifacts, target/audit summary |
| `sync.json` | Offset convention, method, confidence, overlap and fusion uncertainty |
| `cameras.npz` | Camera motion, uncertain intrinsics/scale, table evidence |
| `observations.npz` | Estimated 3D observations with confidence, masks and provenance |
| `retargeted.npz` | Human targets and IK residuals; no block trajectory control |
| `scene_kinematic.xml` | Separate kinematic-reference model |
| `kinematic_reference.mp4` | Explicitly labeled reference playback; **not physics** |
| `scene_initial.xml`, `physics_initial.npz` | Initial physical model and full rollout |
| `scene.xml`, `physics_rollout.npz` | Selected physical model and actual solver states |
| `refinement.json` | Candidate history, validity, selected parameters and improvement |
| `simulation.mp4` | Rendering-only replay of selected physics states |
| `comparison.mp4` | Synchronized source/source/physics panels |
| `simulation.gif`, `comparison.gif` | Optional post-MP4 GIF conversions |
| `tracking.mp4` | Source-space perception diagnostics |
| `world_preview.mp4` | Uncertain common-world diagnostic |
| `metrics.json` | Perception / reconstruction / physics sections and target table |
| `audit.json` | Pass/fail integrity checks with evidence and failure details |
| `logs/block_write_guard.json` | Guard counters, protected addresses and rollout hash |
| `logs/pipeline.log` | Stage progress and errors |
| `cache/` | Keyed stage arrays and rendering receipts |

Never treat kinematic block playback as contact-driven reconstruction.
Observation, retarget and physics artifacts are distinct files.

## Reading metrics and audit

`metrics.json` has the spec 19 sections `perception`, `reconstruction` and
`physics`, plus `targets`. Legacy diagnostic fields remain for compatibility.

Each target records its measured value, units, comparison operator, threshold,
boolean `pass`, status and explanatory detail:

- visible-hand tracking coverage ≥80%;
- mean visible-landmark reprojection ≤15 px;
- block-centroid reprojection ≤20 px;
- physics block trajectory error ≤0.25 block lengths;
- persistent penetration ≤10% of the smallest block dimension.

Unavailable measurements are `null`, `not_evaluable`, and **do not pass**.
The current baseline does not independently measure visible-hand recall or
final-scene visible-landmark/block-centroid reprojection. Frame detection
availability and raw pre-prior lifting residuals are reported as diagnostics,
not silently substituted for these targets.

Trajectory error uses unique observed source-frame/block samples, excluding
held observations. Below the minimum reference coverage, accepted errors are
unavailable and diagnostic means remain separately visible.

Lengths are reported in estimated meters and block-normalized units. Rotation
is in radians; cuboid-symmetry-reduced and raw errors remain separate.
Persistent penetration means depth sustained by a block throughout a 100 ms
window. The maximum instantaneous depth is also retained.

`audit.json` checks artifact existence, block constraints/actuation, runtime
write-guard evidence, finite physics arrays, stability, contact attribution,
artifact separation and unpaired reporting. Contact attribution requires
accumulated excursions greater than 1 cm to have a builder-hand-to-block path
through actual solver contacts in the preceding 0.2 seconds. Table/world and
forearms cannot bridge that path. This is a bounded contact-connectivity
diagnostic, not a proof of exact causal force flow.

The no-NaN audit concerns physics state. Missing visual observations legitimately
contain NaNs with zero confidence.

An audit pass does not establish accurate geometry or shared-room fusion.
Target failure and integrity failure are distinct. Completed, stable diagnostic
runs retain their outputs even when targets or audit checks fail; CLI exit status
primarily reports execution/metadata failure. Read the printed summary and JSON,
not just the exit code.

## Caching and runtime

Re-running the same command in an existing output directory is supported.
Unrelated files are preserved. Input hashes, stage configuration and code
versions key cached arrays; rendering receipts also verify output hashes.

Use `--force` to recompute. Missing or changed models still fail explicitly even
when other stage artifacts exist. Adding guard instrumentation invalidates old
physics caches once; perception caches remain reusable.

Physics-only experimentation, without perception or IK:

```text
python -m interaction_recon.stages_physics outputs/test_1/003 --budget 210 --evaluations 32
```

That command updates physics/refinement artifacts. Re-run the normal CLI afterward
to refresh the final comparison, consolidated metrics and audit.

The integration engineer measured segment_003 M6 end-to-end with cached
perception at **2m48s** before the M7 rendering/audit additions. This is a
historical measurement, not an M7 runtime guarantee. First-run perception,
960×720 rendering, source comparison decoding and GIF conversion add work.
The intended laptop budget remains approximately ten minutes per short pair.
Refinement defaults to a 210-second search budget; wall-time limits can change
the number of candidates despite a fixed random seed.

Media decoding currently holds source RGB frames in memory. Very long or
high-resolution recordings can need substantial RAM; the comparison decodes
the physics movie incrementally.

## Known limitations and benchmark status

- **Monocular metric scale is uncertain.** Learned human geometry, table-height
  assumptions and uncalibrated FOV priors are not millimeter-accurate measurement.
- **Cross-view fusion is unverified for this archive.** The two segment_003 streams
  appear to show different rooms, tables and people. The pipeline produces a
  role-composed common scene with low-confidence flags, not a falsely certified
  shared interaction. Guider block observations remain separate from the
  builder-only physical block set.
- **Absolute trajectory accuracy is above target.** The integrating engineer's
  M6 segment_003 result improved position error from approximately 0.165 m
  (0.83 BL) to 0.158 m (0.79 BL), and rotation from 0.62 to 0.59 rad.
  Position improved 4.3% and total loss 4.4%: measurable M6 improvement, but
  **the ≤0.25 BL absolute target was not met**. These are reported historical
  measurements, not numbers hard-coded into evaluation.
- Egocentric hand detection and ownership/handedness remain incomplete.
  Color/PCA forearm fallbacks do not fabricate observed fingers.
- OpenCV camera translation can drift or remain unobservable. Curved/clipped
  tables can require explicitly uncertain priors.
- Block fragmentation, geometric identity consolidation, inferred height and
  hand-depth contact priors are uncertain. Post-prior proximity is not
  independent evidence of real contact.
- Contact switching makes inverse physics non-monotonic and parameters only
  partially identifiable. A similar trajectory need not recover unique friction.
- Automatic target evaluation is deliberately incomplete where independent
  evidence is unavailable. Those targets are reported as not evaluable.
- Auto-framing may make the table narrower than 70% when complete actor bounds
  demand it; it does not clip actors to manufacture a tighter-looking scene.

## Tests

```text
python -m pytest
```

Tests cover discovery and timing, geometry/retargeting, block-write guards,
contact-driven motion, refinement, synthetic comparison layout, bundled-FFmpeg
GIF conversion, injected block-actuation violations and target thresholds.

The M7 additions have not been executed in the integration environment as part
of this code delivery; no new passing-test count or M7 benchmark is claimed.
