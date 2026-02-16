# SUMMARY

## 2026-02-16 | Assistant: Codex (GPT-5)

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
