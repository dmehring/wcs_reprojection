# SUMMARY

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

## 2026-02-19 01:00 (approx) | Assistant: Gemini 3 Flash

### Gaussian Generation and Moments
- **Implemented** a robust `generate_gaussian_xy` function supporting the `data[x, y]` convention.
- **Enforced** `sigma_a >= sigma_b` to concretely define major vs. minor axes and resolve orientation ambiguity.
- **Developed** a moment-based analyzer to recover `theta_math` (pixel-space angle) using second-order central moments ($\mu_{20}, \mu_{02}, \mu_{11}$).

### WCS and Parity Handling
- **Configured** a test WCS using the **PC + CDELT** convention to mirror NRAO/CASA metadata standards.
- **Handled** **East-Left** parity ($CDELT1 < 0$) by utilizing the determinant of the WCS `pixel_scale_matrix`.
- **Corrected** the "North" reference vector calculation to use `arctan2(dy, dx)` for the `[x, y]` indexing order, ensuring North ($+Dec$) is correctly identified in pixel space.

### Astronomical Position Angle (PA) Logic
- **Derived** the final transformation to convert pixel-space `theta_math` to Astronomical PA (North through East).
- **Refined** the subtraction order to `(north_pixel_angle - theta_math) * parity` to align with the clockwise/counter-clockwise shifts inherent in astronomical projections.
- **Implemented** the **Radio Astronomy Standard** range shift:
  - Mapped the standard $[0, 180^\circ)$ PA to the preferred radio range of $(-90^\circ, 90^\circ]$.

### Visualization and Verification
- **Validated** the `data[x, y]` to `imshow` transition by applying `.T` (transpose) and `origin='lower'` to ensure visual alignment between the array and the WCS coordinate grid.
- **Created** a verification suite to check cardinal and ordinal directions (North, North-East, East) against expected PA values.

### dmehring note
- added test implementtaion to notebooks/wcs_reprojection_examples.ipynb for now. should move to
  notebook of its own.

## 2026-02-19 22:01 UTC | Assistant: Codex (GPT-5)

### WCS Builder Documentation
- Fully documented `build_wcs_from_xradio` in `wcs_reproject.py` with a complete standards-compliant docstring.
- Added explicit parameter semantics for:
  - frame and reference-direction overrides,
  - supported frame choices and resulting axis-type behavior,
  - coordinate override handling and expected units.
- Documented return structure (`_WCSBuildResult`), core invariants/assumptions (radian inputs, degree-based FITS-WCS output, 2D celestial scope), and key failure modes (`KeyError`, `ValueError`, `IndexError`).

## 2026-02-19 22:25 UTC | Assistant: Codex (GPT-5)

### Notebook Cell Extraction
- Moved the first cell from `notebooks/wcs_reprojection_examples.ipynb` into a new notebook: `notebooks/plot_astro_images_with_matplotlib.ipynb`.
- Left the source notebook with the remaining cells in original order and preserved notebook metadata/format fields in both files.

## 2026-02-20 00:00 UTC | Assistant: Codex (GPT-5)

### Notebook Fix: `build_wcs_from_xradio` Demo
- Fixed `NameError` in `notebooks/demonstrate_build_wcs_from_xradio.ipynb` by defining `sky_data` before constructing the demo `DataArray`.
- Constructed `sky_data` with an explicit shape from the selected dimensions (`time`, `frequency`, `polarization`, `l`, `m`) to keep the example deterministic and self-contained.
- Cleared stale notebook execution output/error traceback so the notebook is committed in a clean state.

## 2026-02-20 03:56 UTC | Assistant: Codex (GPT-5)

### WCS Builder Compatibility Fix
- Fixed `build_wcs_from_xradio` in `wcs_reproject.py` to avoid writing `wcs.wcs.naxis1` / `wcs.wcs.naxis2`, which are not valid attributes on `astropy.wcs.Wcsprm`.
- Updated the function to store output geometry on the high-level `WCS` object using:
  - `wcs.pixel_shape = (naxis1, naxis2)` and
  - `wcs.array_shape = (naxis2, naxis1)`.
- Verified with a direct local sanity check that WCS construction now succeeds and reports expected `ctype`, `pixel_shape`, and `array_shape`.

## 2026-02-20 05:25 UTC | Assistant: Codex (GPT-5)

### Notebook Example Addition: XRADIO Image + WCS Plot
- Added a new code cell to `notebooks/plot_astro_images_with_matplotlib.ipynb` that:
  - creates a synthetic XRADIO image with `make_empty_sky_image`,
  - constructs a `DataArray` carrying XRADIO-style metadata,
  - builds WCS via `build_wcs_from_xradio(..., dim_a="l", dim_b="m")`,
  - extracts a 2D plane and plots it using `plotting.generate_astro_plot`.
- Included a simple off-center point source in the synthetic plane so the rendered plot has visible structure.

## 2026-02-20 06:39 UTC | Assistant: Codex (GPT-5)

### Notebook Corner-Coordinate Validation Cell
- Appended a new final code cell to `notebooks/plot_astro_images_with_matplotlib.ipynb` that validates Galactic corner coordinates by comparing:
  - values taken from the reprojected dataset coordinate arrays (`galactic_longitude`, `galactic_latitude`), and
  - Astropy-transformed corner RA/Dec values from the source dataset (`xds`) using `SkyCoord(...).galactic`.
- The cell prints per-corner residuals (`dlon_arcsec`, `dlat_arcsec`) so frame/world-coordinate consistency is directly checkable.
- The implementation uses existing `corners_xy` when present, with a fallback to image corners.

## 2026-02-20 06:54 UTC | Assistant: Codex (GPT-5)

### Notebook Fix: Valid FK5->Galactic Corner Comparison
- Replaced the previous same-index corner comparison cell in `notebooks/plot_astro_images_with_matplotlib.ipynb` with a physically correct comparison pipeline:
  - source corner pixel -> source FK5 RA/Dec (from source world-coordinate maps),
  - FK5 RA/Dec -> Astropy Galactic (`glon/glat`),
  - Astropy Galactic world -> target pixel (`wcs_world2pix` on the transformed dataset WCS),
  - bilinear sample of target `galactic_longitude/galactic_latitude` at mapped target pixels.
- The new output prints source pixel, mapped target pixel, both Galactic coordinate values, and arcsecond residuals (`dlon_arcsec`, `dlat_arcsec`).
- Cleared stale outputs from the replaced cell.

## 2026-03-02 07:35 UTC | Assistant: Codex (GPT-5)

### Full Function Documentation Pass (`wcs_reproject.py`)
- Added comprehensive docstrings for all helper functions that previously lacked complete documentation.
- Expanded coverage to include parameter semantics, return values, supported choice enumerations (for example reprojection method and frame mapping), and behavior/invariant notes.
- Documented internal metadata/coordinate-management helpers, including stale world-coordinate alias cleanup behavior in output assembly.
- Added a docstring for the nested per-plane reprojection helper inside `_reproject_dataarray` to keep internal behavior explicit.
