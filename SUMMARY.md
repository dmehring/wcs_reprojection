# SUMMARY

## 2026-02-17 01:47 | Assistant: Codex (GPT-5)

### Frame Conversion and Axis Handling
- Fixed reprojection axis-order handling in `wcs_reproject.py` so frame-converted outputs are spatially correct and no longer appear unintentionally unrotated.
- Confirmed `keep_grid=False` produces Galactic world-coordinate axes aligned with image edges.
- Preserved robust peak world-position consistency checks for both `keep_grid=False` and `keep_grid=True`.

### Beam PA Sign Correction
- Corrected beam PA update sign during frame conversion:
  - Beam `pa` in `BEAM_FIT_PARAMS_*` is now updated with the opposite sign of the local frame-basis angle so PA remains consistent with displayed source orientation.
- Verified Example 3 now reports negative beam PA delta for FK5 -> Galactic in the shown setup, matching plot convention expectations.

### Example 3 Verification Upgrades
- Reworked Example 3 notebook checks to use an off-center point source for position/grid tests and a Gaussian source for orientation tests.
- Added explicit user-facing checks for:
  - Peak world-position consistency (Astropy-transformed input vs output world coords).
  - Galactic-grid edge parallelism (`keep_grid=False`) via cross-axis ratios.
  - Beam/source major-axis PA consistency with sign-aware reporting in plot convention.
- Updated Example 3 printouts to avoid ambiguous sign interpretation and provide a clear PASS/CHECK summary.

### Regression Tests
- Added `tests/test_reproject_to_frame_galactic.py` with focused regression coverage:
  - peak world-position consistency for both keep-grid modes,
  - Galactic grid parallel-to-edge behavior for `keep_grid=False`,
  - beam PA change consistency with measured source major-axis rotation.
- Ran `flake8` clean on touched Python files and reformatted updated test code with `black`.

## 2026-02-16 16:21 | Assistant: Codex (GPT-5)

### Packaging and Environment
- Added notebook-focused optional dependencies in `pyproject.toml` for reproducible example runs.
- Installed and validated development tools: `black`, `flake8`, and `ipython`.

### Reprojection API and Behavior
- Extended `reproject_to_match` to support data-group based reprojection for XRADIO datasets:
  - Added `data_group` (default `"base"`).
  - Reprojects all group variables containing spatial dims.
  - Preserves backward compatibility with `data_var` fallback.
- Fixed target-grid handling so reprojection output shape/coords match the target grid when source and target sizes differ.
- Added support for target datasets that define grid/WCS metadata without an explicit `SKY` variable.

### Frame Conversion Metadata
- Improved `reproject_to_frame` metadata handling:
  - Added frame-consistent world-coordinate regeneration.
  - Added optional retention of input world coordinates via `keep_input_world_coords=True`.
  - Added conditional behavior so world coords are only generated when input world coords exist.

### Beam Consistency
- Added beam position-angle rotation during frame changes:
  - `BEAM_FIT_PARAMS_*` `pa` values are rotated to stay consistent with the new frame basis.

### Notebook Improvements
- Updated examples to use XRADIO image loading and data-group reprojection patterns.
- Added stronger numerical checks, especially in Example 2:
  - Explicitly distinguishes raw pixel sum from physically meaningful flux-density quantities.
  - Verifies integrated flux-density consistency for `Jy/beam` under pixel-scale changes.
- Updated plotting helper for astronomy-friendly visualization:
  - Angular offset axes in arcsec.
  - `l` horizontal, `m` vertical.
  - `l` decreases left-to-right.
- Updated Example 2 to use an off-center source for better generality.
- Updated Example 3 to demonstrate frame-world-coordinate behavior and optional retention of original axes.
- Ensured notebook outputs are cleared before commit.

### Documentation and Workflow
- Added and refined `AGENTS.md` with project strategy, coding/testing conventions, and assistant personalization.
- Updated README install/usage guidance to align with XRADIO-first workflows.
