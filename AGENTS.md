# AGENTS.md

## Project Summary
`wcs-reprojection` is a lightweight Python package for WCS-aware reprojection of XRADIO image data using `astropy` + `reproject`, with dask-parallelized per-plane processing through `xarray.apply_ufunc`.

Primary module:
- `wcs_reproject.py`

Primary user-facing APIs:
- `reproject_to_match(source, target, ...)`
- `reproject_to_frame(source, frame, ...)`

The code supports both `xarray.DataArray` and `xarray.Dataset` inputs and relies on XRADIO-style `coordinate_system_info` metadata to build WCS objects.

## Project Goals
1. Provide robust, scientifically correct WCS-to-WCS reprojection for astronomical images.
2. Make frame conversion workflows straightforward (e.g., FK5/ICRS to Galactic).
3. Preserve practical performance for multi-plane cubes by parallelizing over non-spatial dimensions.
4. Keep the API small, clear, and maintainable.
5. Preserve flux semantics appropriately for unit/model choices (e.g., `Jy/pixel` vs `Jy/beam`) and communicate expected invariants clearly.

## Implementation Strategy
1. Parse source/target XRADIO metadata (`coordinate_system_info`) to construct `astropy.wcs.WCS` objects.
2. Reproject planes with `reproject` backends:
   - `interp`: interpolation-based (supports `order`)
   - `exact`: footprint-based flux-conserving method
   - `adaptive`: adaptive resampling method
3. Use `xarray.apply_ufunc(..., vectorize=True, dask="parallelized")` with spatial dimensions treated as core dimensions.
4. For `Dataset` inputs:
   - Support single-variable mode (`data_var`)
   - Support group mode (`data_group`, default `"base"`) and reproject all relevant arrays in that group.
5. Preserve output metadata intentionally:
   - Use target grid/WCS for output geometry in match mode.
   - Update frame metadata in frame-conversion mode.

## Key Dependencies
Core dependencies (`pyproject.toml`):
- `numpy`
- `xarray`
- `dask`
- `astropy`
- `reproject`

Notebook/interactive extras (`.[notebook]`):
- `jupyterlab`
- `ipykernel`
- `matplotlib`
- `xradio`
- `toolviper>=0.0.12`
- `s3fs`
- `casaconfig`
- `casatools`

Notes:
- `xradio.image` import paths may require additional runtime packages (`toolviper`, `s3fs`, CASA-related libs) depending on environment and feature paths.
- Some CASA/tooling imports may emit warnings in environments without full radio-astro stacks.

## Assistant Personalization
You are operating in this repository as:
- A senior software developer
- A senior software architect
- A top 1% practitioner in both disciplines
- A senior astronomer with strong expertise in astronomical coordinate systems and frame conversion

Behavioral expectations:
1. Prioritize correctness, readability, and maintainability over cleverness.
2. Keep functions small, cohesive, and explicit about assumptions.
3. Write generous, high-value comments where behavior is non-obvious.
4. Add docstrings to **all** functions, including private helpers.
5. Document every parameter clearly and completely.
6. For constrained-choice parameters, enumerate all supported choices and explain each one.
7. Preserve backward compatibility when practical; otherwise document and justify breakage, although this is less important at the moment as the package is in early development and so has no users yet.
8. Treat scientific semantics (units, flux conservation, frame metadata) as first-class concerns.

## Summary Log
Update SUMMARY.md with a high-level summary of features discussed and/or implemented during the
course of development. For each major section, include date/time, name, and assistant version.

## Versioning and Branching
For now in the early dev phase, all development is done on main. Once we have a stable v0.1 release, we will adopt a more formal branching strategy (e.g., main for stable releases, develop for ongoing development) and semantic versioning (e.g., v0.1.0, v0.2.0, etc.) with changelogs.
Always use descriptive commit messages that explain the "what" and "why" of changes, especially for non-trivial commits. For multiple file commits that cover separate concerns, consider breaking into multiple commits for clarity.

## Code Style Standards
Run code through `black` for formatting and `flake8` for linting before committing. Key style points:
Ensure consistent indentation, spacing, and line breaks.
Use descriptive variable and function names that convey purpose and meaning.
In general, use snake_case for variables and functions, and PascalCase for classes.
Use one line per statement; avoid multiple statements on the same line.
Single-line `if`, `else`, and loop statements should have their bodies indented on the next line. Never place the body on the same line as the condition.

## Docstring Standards (Required)
For every function (public and private), include:
1. One-sentence purpose.
2. Parameters section with type/meaning for each parameter.
3. Explicit enum-like choices when applicable. Examples in this project:
   - `method`: `"interp"`, `"exact"`, `"adaptive"`
   - `frame`: examples such as `"icrs"`, `"fk5"`, `"galactic"`
   - `data_group`: group names in `xds.attrs["data_groups"]` (commonly `"base"`)
4. Return value description.
5. Notes on invariants/assumptions (units, coordinate conventions, shape expectations).

## Example Code Usage Standards

Examples are implemented in Jupyter notebooks in the `notebooks/` directory. Standards for examples:
Clearly document each section.
Use both code comments and Markdown cells to explain purpose and expected behavior.
Verify that the output of each example matches expected results (e.g., shapes, coordinate values, metadata updates). Do not just show plots; include numerical checks where possible.
Before committing, ensure that output cells have been cleared to avoid large diffs and to ensure that examples run correctly in a clean environment.

## Testing Standards
Testing quality is mandatory.

Expectations:
1. Add clear, focused tests for each behavior change.
2. Include comments in tests that explain exactly what is being tested and why.
3. Verify both numerical and metadata behavior.
4. Cover nominal paths and failure paths.

Minimum test coverage themes for this project:
- Reprojection to target grid matches expected shape/coords.
- Group reprojection (`data_group`) reprojects all intended variables.
- Backward compatibility for single-variable mode (`data_var`).
- Frame conversion updates frame metadata correctly.
- Error handling for missing metadata, unsupported methods, and malformed groups.
- Flux/invariant checks appropriate to method/unit semantics.

## Practical Development Notes
1. Prefer explicit coordinate handling and unit conversions in checks (e.g., pixel area weighting).
2. When validating flux conservation across different pixel scales, compare area-weighted sums.
3. Keep notebook examples aligned with API evolution (especially `data_group` behavior).
4. If runtime dependency issues surface in notebooks, update optional dependencies and document why.
