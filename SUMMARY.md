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

## 2026-03-02 08:02 UTC | Assistant: Codex (GPT-5)

### `reproject_to_frame` `keep_grid` Docstring Clarification
- Expanded the `keep_grid` parameter documentation in `wcs_reproject.py` to explicitly describe reference-direction and `l/m` zero-point behavior in both modes.
- Documented that `keep_grid=True` reuses the input pixel grid while still updating frame/reference-direction metadata.
- Documented that `keep_grid=False` rebuilds a centered same-size/same-spacing grid where `l=0, m=0` is at the image midpoint.

## 2026-03-02 08:11 UTC | Assistant: Codex (GPT-5)

### Removed Optional World-Coordinate Retention/Skipping in `reproject_to_frame`
- Removed `update_world_coords` and `keep_input_world_coords` from the `reproject_to_frame` API.
- Made world-coordinate replacement unconditional in frame reprojection so output world coordinates are always regenerated from the output WCS and target frame.
- Removed the legacy `keep_input` branch from `_replace_world_coords`; stale canonical/alias world-coordinate arrays are dropped before writing fresh frame-consistent coordinates.

## 2026-03-02 08:23 UTC | Assistant: Codex (GPT-5)

### Added Pixel-Consistent Original-Frame Coordinate Output for `reproject_to_frame`
- Added new `include_original_world_coords` parameter to `reproject_to_frame`.
- When enabled, output now includes `original_*` world-coordinate arrays for the source frame (for example `original_right_ascension`, `original_declination`) computed by transforming each output-pixel world coordinate from the target frame back into the original frame.
- This guarantees one-to-one pixel-wise consistency between target-frame and original-frame coordinate grids on the output image.
- Added regression coverage to verify transformed original-frame coordinates match Astropy frame transforms at all output pixels within tight tolerance.

## 2026-03-02 22:48 UTC | Assistant: Codex (GPT-5)

### Added Additional Easy Frame Support (Ecliptic Family)
- Extended frame mapping in `wcs_reproject.py` to support non-observer-dependent ecliptic frame conversions for:
  - `geocentrictrueecliptic`
  - `geocentricmeanecliptic`
  - `barycentrictrueecliptic`
  - `barycentricmeanecliptic`
- Added ecliptic coordinate naming support (`ecliptic_longitude`, `ecliptic_latitude`) and WCS CTYPE mapping (`ELON/ELAT`).
- Kept support focused on easy/non-observer-dependent frame families; supergalactic mapping via WCS CTYPE was not retained due `reproject`/Astropy pixel-to-pixel class mismatch in cross-frame reprojection.
- Added regression tests to verify ecliptic coordinate emission and pixelwise consistency of `original_*` coordinates when `include_original_world_coords=True`.

## 2026-03-02 23:12 UTC | Assistant: Codex (GPT-5)

### Explicit `reproject_to_frame` Supported Frame List + Case-Insensitive Note
- Updated `reproject_to_frame` docstring to explicitly enumerate supported `frame` strings:
  - `icrs`, `fk5`, `fk4`, `fk4noeterms`, `galactic` (`gal`),
  - `geocentrictrueecliptic`, `geocentricmeanecliptic`,
  - `barycentrictrueecliptic`, `barycentricmeanecliptic`.
- Documented that supported frame strings are case-independent.
- Normalized `frame` to lowercase in `reproject_to_frame` implementation to enforce case-insensitive behavior consistently.
- Updated `build_wcs_from_xradio` frame-override documentation to note case-insensitive frame handling.

## 2026-03-02 23:23 UTC | Assistant: Codex (GPT-5)

### `reproject_to_frame` Data-Group Support + Fallback to `data_var`
- Added `data_group` support to `reproject_to_frame`, matching `reproject_to_match` behavior intent for Dataset inputs.
- Implemented Dataset group-frame reprojection path that reprojects all spatial variables resolved from the selected group to the requested frame.
- Added explicit fallback behavior: when group metadata is missing or cannot be resolved to reprojectable spatial variables, `reproject_to_frame` now falls back to single-variable mode using `data_var`.
- Added regression tests validating:
  - all group spatial variables are reprojected in group mode, and
  - unresolved/missing group selection falls back to `data_var`.

## 2026-03-02 23:58 UTC | Assistant: Codex (GPT-5)

### Docstring Clarification: `order` Supports String and Integer Forms
- Updated interpolation-order documentation in `wcs_reproject.py` for all relevant functions to explicitly state accepted string and integer equivalents for `method="interp"`.
- Added explicit mapping:
  - `"nearest-neighbor"` <-> `0`
  - `"bilinear"` <-> `1`
  - `"biquadratic"` <-> `2`
  - `"bicubic"` <-> `3`

## 2026-03-03 00:10 UTC | Assistant: Codex (GPT-5)

### API Simplification: Removed `order`, Expanded `method` Selector
- Removed the `order` parameter from `reproject_to_match` and `reproject_to_frame` (and internal helpers) since this is pre-release and no backward compatibility is required.
- Consolidated interpolation selection into `method`:
  - interpolation IDs: `0`, `1`, `2`, `3`
  - interpolation names: `"nearest-neighbor"`, `"bilinear"`, `"biquadratic"`, `"bicubic"`
  - non-interp methods: `"interp"` (defaults to bilinear), `"exact"`, `"adaptive"`
- Updated docstrings and type hints to reflect method-only selector behavior and explicit algorithm descriptions.
- Added regression tests verifying selector equivalence (`"interp"` == `1` == `"bilinear"`) and invalid-selector failures.

## 2026-03-03 00:14 UTC | Assistant: Codex (GPT-5)

### Method Selector Simplification: Default `bilinear`, Removed `"interp"` Alias
- Changed default `method` for `reproject_to_match` and `reproject_to_frame` to `"bilinear"` so interpolation behavior is explicit by default.
- Removed support for the `"interp"` method alias.
- Kept supported `method` selectors as:
  - interpolation IDs: `0`, `1`, `2`, `3`
  - interpolation names: `"nearest-neighbor"`, `"bilinear"`, `"biquadratic"`, `"bicubic"`
  - non-interpolation methods: `"exact"`, `"adaptive"`
- Updated tests to validate default-is-bilinear behavior and selector equivalence.

## 2026-03-03 00:25 UTC | Assistant: Codex (GPT-5)

### Consistency Refactor: Unified `data_group` Fallback in Match/Frame APIs
- Refactored shared Dataset group-selection setup into `_resolve_reproject_group_vars_with_fallback(...)`.
- Updated `reproject_to_match` to use the same fallback behavior as `reproject_to_frame`: if group metadata is missing or group resolution is invalid/unreprojectable, fallback to `data_var` single-variable mode.
- Updated `reproject_to_match` docstring to document fallback behavior explicitly.
- Added new regression tests in `tests/test_reproject_to_match_data_group.py` verifying:
  - all spatial group vars are reprojected when a valid group is provided, and
  - unresolved group definitions fall back to `data_var` mode.

## 2026-03-03 00:30 UTC | Assistant: Codex (GPT-5)

### Open Item: FLAG Array Reprojection Policy
- Noted unresolved follow-up: define and implement explicit boolean FLAG-array reprojection semantics (boolean-in/boolean-out), including policy choice (`any`/`majority`/`all`), coverage behavior, and tests.

## 2026-03-03 05:11 UTC | Assistant: Codex (GPT-5)

### New Notebook: Minimal `reproject_to_match` Same-Frame Validation
- Added `notebooks/reproject_to_match_minimal_same_frame.ipynb` with a focused same-frame (FK5->FK5) reprojection example.
- Example configuration enforces:
  - source pixel scale = `1.4x` target pixel scale,
  - source reference-direction offset = `(20.5, 20.5)` source pixels,
  - two Gaussian sources with different sizes placed in opposite quadrants.
- Included explicit numeric checks showing:
  - target/output world-coordinate direction grids are identical, and
  - target/output source-peak consistency is `<= 0.5` pixels.

## 2026-03-03 05:15 UTC | Assistant: Codex (GPT-5)

### Notebook UX Rewrite: `reproject_to_match` Minimal Example
- Reworked `notebooks/reproject_to_match_minimal_same_frame.ipynb` for user-facing readability.
- Added step-by-step markdown explanations between code cells and expanded inline code comments for each stage of setup, reprojection, and validation.
- Removed malformed quoted markdown artifacts and replaced with proper Markdown list/section formatting.

## 2026-03-03 05:23 UTC | Assistant: Codex (GPT-5)

### Notebook Plotting Additions: Source, Target, and Output
- Updated `notebooks/reproject_to_match_minimal_same_frame.ipynb` to include astronomer-friendly plots for:
  - source image,
  - target image,
  - reprojection output image.
- Used `plotting.generate_astro_plot(...)` with WCS from `build_wcs_from_xradio(...)` so orientation follows expected astronomy conventions.
- Preserved clean notebook state by clearing execution outputs after edits.

## 2026-03-03 05:31 UTC | Assistant: Codex (GPT-5)

### Notebook Proof Cell: Reference Shift in Source-Pixel Units
- Added an explicit verification section to `notebooks/reproject_to_match_minimal_same_frame.ipynb` proving the stated `(20.5, 20.5)` reference-direction shift.
- New cell computes source-target reference-direction deltas from `coordinate_system_info`, converts them to source-pixel units using source-axis pixel scale, prints intermediate quantities, and asserts agreement with `(20.5, 20.5)`.

## 2026-03-03 05:41 UTC | Assistant: Codex (GPT-5)

### Notebook UX Tweak: Immediate Source Plot After Gaussian Construction
- Updated `notebooks/reproject_to_match_minimal_same_frame.ipynb` so section **4) Populate the source image with two Gaussians** now ends by plotting the source image immediately after population.
- Plot uses `generate_astro_plot(...)` with WCS from `build_wcs_from_xradio(...)` for astronomy-friendly orientation.

## 2026-03-03 05:48 UTC | Assistant: Codex (GPT-5)

### Notebook Plot Labeling Update: Explicit RA/Dec Axis Labels
- Updated plotting cells in `notebooks/reproject_to_match_minimal_same_frame.ipynb` to explicitly set axis labels to `Right Ascension` and `Declination`.
- This replaces generic WCSAxes auto-label text (for example `pos.eq.ra` / `pos.eq.dec`) for clearer user-facing presentation.

## 2026-03-03 05:58 UTC | Assistant: Codex (GPT-5)

### Notebook Validation Addition: Source vs Output Peak World Coordinates
- Added a new final validation section to `notebooks/reproject_to_match_minimal_same_frame.ipynb` that compares FK5 world coordinates at source-image peaks vs output-image peaks for both Gaussian sources.
- The new cell computes spherical small-angle separations in arcseconds, prints per-source and max separation, and asserts the max separation is within a defined tolerance.

## 2026-03-03 06:18 UTC | Assistant: Codex (GPT-5)

### Notebook Fix: WCS-Consistent Source/Output Peak World-Coordinate Check
- Updated Validation C in `notebooks/reproject_to_match_minimal_same_frame.ipynb` to compute source/output peak world coordinates via `build_wcs_from_xradio(...).wcs_pix2world(...)` on both sides.
- This avoids mixing coordinate arrays with potentially different derivation paths and aligns the validation with the WCS basis used by reprojection.
- Kept detailed per-source RA/Dec prints and delta reporting for debugging visibility.

## 2026-03-03 06:21 UTC | Assistant: Codex (GPT-5)

### Notebook Clarity Update: Expanded Inline Comments in Final Validation Cell
- Reworked the final world-coordinate peak-comparison cell in `notebooks/reproject_to_match_minimal_same_frame.ipynb` with explicit sectioned inline comments.
- Added clear step labels for: data/WCS setup, peak selection, world-coordinate conversion, angular-difference math, per-source reporting, and final pass/fail criterion.

## 2026-03-03 06:35 UTC | Assistant: Codex (GPT-5)

### Notebook Refactor: Target as Coordinate Template (Zero Intensity)
- Refactored `notebooks/reproject_to_match_minimal_same_frame.ipynb` so target `SKY` values are explicitly all zeros and used only as a coordinate/WCS template.
- Removed reliance on target intensity structure for scientific comparisons.
- Updated validations to focus on source/output consistency:
  - mapped source-peak vs output-peak pixel agreement,
  - source/output peak world-coordinate agreement,
  while retaining target/output world-grid identity checks.

## 2026-03-03 06:39 UTC | Assistant: Codex (GPT-5)

### Notebook Clarification: RA Handedness and Shift Sign
- Added section `3c) RA handedness note` to `notebooks/reproject_to_match_minimal_same_frame.ipynb`.
- New cell computes source-reference direction location in target pixel coordinates via WCS and prints signed pixel deltas from image center.
- Explicitly documents that `CDELT1 < 0` implies RA increases to the left, so horizontal display-axis sign can appear flipped relative to local-offset metadata sign.

## 2026-03-03 06:42 UTC | Assistant: Codex (GPT-5)

### Notebook Fix: Signed vs Magnitude Pixel Shift in Reference-Offset Proof
- Updated section `3b` in `notebooks/reproject_to_match_minimal_same_frame.ipynb` to distinguish signed pixel shifts from magnitude shifts.
- Added explicit note that negative signed `l` shift is expected when `l` axis spacing is negative (RA-left handedness).
- Updated assertion to validate magnitude offsets `(20.5, 20.5)` rather than forcing positive signed offsets.

## 2026-03-03 06:46 UTC | Assistant: Codex (GPT-5)

### Notebook Bugfixes: Validation-B Scalar Cast + Remove Zero-Target Plot
- Fixed `TypeError` in Validation B by coercing `wcs_world2pix` outputs to Python floats before passing to `peak_in_window(...)`.
- Updated plotting sections to remove the zero-valued target image plot (target remains a coordinate template only).
- Kept source and output plots plus all validation logic intact.

## 2026-03-03 06:52 UTC | Assistant: Codex (GPT-5)

### Notebook Validation Clarification: Pixel Diagnostic vs World Invariance
- Updated Validation B in `notebooks/reproject_to_match_minimal_same_frame.ipynb` to be informational only (removed hard `<= 0.5` assert).
- Added explicit comments that pixel-space peak offsets are secondary diagnostics under resampling, while world-coordinate peak agreement (Validation C) is the primary physical preservation check.

## 2026-03-03 07:01 UTC | Assistant: Codex (GPT-5)

### Notebook Bugfix: Validation-B String Literal + Clean Scalar Printing
- Fixed unterminated string literal in Validation B table-report cell (`\n` print line).
- Normalized WCS diagnostic prints in section `3c` to plain Python floats (removed `array(...)` / `np.float64(...)` noise from displayed tuples).
- Syntax-checked all notebook code cells after edits and kept outputs cleared.

## 2026-03-03 07:16 UTC | Assistant: Codex (GPT-5)

### Notebook Nomenclature Standardization + World-Check Tolerance Update
- Standardized notebook terminology to use:
  - `input` for the original image,
  - `template` for the coordinate-template image,
  - `output` for the reprojected result.
- Updated variable names and markdown text across `notebooks/reproject_to_match_minimal_same_frame.ipynb` to match that nomenclature.
- Updated world-coordinate peak agreement tolerance to half the larger pixel size:
  - `0.5 * max(input_cell_arcsec, template_cell_arcsec)`.
