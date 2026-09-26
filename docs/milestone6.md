# Milestone 6: bounded inverse physics

Normal reconstruction now runs the cached refinement stage after the initial
contact rollout. To rerun physics/refinement without decoding media, perception,
world reconstruction, or IK:

```powershell
python -m interaction_recon.stages_physics outputs/test_1/003 --budget 210 --evaluations 32 --top-k 3
```

`--force` on this standalone command recomputes physics and refinement only.
Changing refinement configuration invalidates the refine cache, not the initial
physics cache or upstream caches.

## Artifacts

- `physics_initial.npz`, `scene_initial.xml`: unrefined simulation.
- `refinement.json`: initial score, every completed/failed/timed-out candidate,
  physical parameters, normalized parameters, loss terms, validity reasons,
  selected candidate, and absolute/relative improvement.
- `physics_rollout.npz`, `scene.xml`, `simulation.mp4`: selected full-resolution
  rollout. If no valid candidate exists, the initial rollout remains available
  explicitly as an uncertified diagnostic fallback.
- `metrics.json`: `physics_initial`, `physics_refined`,
  `refinement_improvement`, and `m6_pass`.

The selected rollout minimizes total loss among valid full-resolution
candidates, including the initial rollout. M6 passes only when the selected
valid candidate also measurably reduces the accepted full-clip position error:
more than both 0.1 mm and 0.1% by default. These are numerical improvement
thresholds, not claims of monocular metric accuracy.

## Attribution and symmetry

Movement attribution uses the union of solver contact edges in the preceding
0.2 seconds. A shortest path may traverse blocks but not the table, world,
guider, or builder forearms. Chain lengths count contact edges. This is a
window-connectivity test, not a reconstruction of temporally ordered force flow.

A general rectangular cuboid has four distinct proper rotational symmetries.
The implementation evaluates their eight signed quaternion representatives.
Reflections are excluded from the rotation geodesic; 90-degree axis exchanges
are not silently accepted for unequal box dimensions. Raw rotation error is
retained separately.

## Runtime and limitations

- Search defaults to a 210-second budget, leaving approximately 30 seconds of
  a four-minute target for the initial simulation, export, and final rendering.
  The four-minute total is a target, not a guaranteed hardware-independent
  limit. Actual stage time and target compliance are reported.
- Candidate stepping checks the deadline every 32 substeps. Model compilation,
  diagnostics, export, and rendering are not forcibly interrupted.
- The short search scores the most-motion observed builder window but simulates
  the original prefix to it. It never initializes blocks from later reference
  poses to make a short-window fit easier.
- Search is serial and seeded. Completed evaluations are deterministic for
  fixed inputs and runtime versions; wall-time cutoff can change their count.
- MuJoCo's default refsafety clamps contact time constants below twice the
  timestep. Requested values are preserved and the effective lower bound is
  recorded, including during 0.004-second coarse rollouts.
- Contact timing compares uncertain observed 3D proximity onsets with simulated
  hand contacts. It excludes held observations but cannot remove uncertainty
  already introduced by upstream metric/depth priors.
- Missing hand/proximity evidence is counted and reported, not fabricated.
- A short search may fail to improve an unidentifiable or geometrically
  inaccurate reconstruction. Such failure does not become an M6 pass.
- The new tests and real benchmark have not been executed in this response
  environment; no passing-test count or measured benchmark improvement is claimed.
