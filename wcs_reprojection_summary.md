# WCS Reprojection Summary (XRADIO Images)

## Use Case
Regrid one XRADIO image to match the coordinate system of another image. This can be:
- Same sky frame (ICRS -> ICRS) but different pixel scales / offsets, or
- Different sky frames (ICRS -> Galactic), which implies rotation and full WCS transformation.

Typical data characteristics:
- XRADIO Zarr datasets with `SKY` data variable.
- Unit semantics stored in `SKY.attrs["units"]`, often `Jy/pixel` or `Jy/beam`.
- Coordinate metadata in `ds.attrs["coordinate_system_info"]`.

## Key Principle
This is a **WCS-to-WCS reprojection** problem, not just interpolation on numeric coordinate arrays.
A proper solution must transform the full sky-frame mapping (including rotation and projection),
then resample onto the target grid.

## Two Cases
### 1) Same frame, no rotation/distortion
If both images share the same frame (ICRS -> ICRS) and are axis-aligned with regular grids,
standard grid interpolation can work:
- `xarray.interp` or `xesmf` on `l/m` or `ra/dec` coordinates.

### 2) Rotation and/or frame change
If the target grid is rotated or the sky frame changes (ICRS -> Galactic),
use a WCS-aware reprojection method. Plain interpolation on numeric arrays is insufficient.

## Method Choice by Units
### Interpolation-based reprojection
- Treats the source as a continuous field sampled at pixel centers.
- Preserves local values but **does not conserve total flux** when pixel areas change.
- Appropriate for `Jy/beam` or intensity-like use cases (with beam metadata checks).

### Flux-conserving reprojection
- Integrates source signal over target pixel footprints.
- **Preserves integrated flux**, required for `Jy/pixel` semantics.

## Recommended Workflow
1. Build WCS objects for source and target using XRADIO metadata (`coordinate_system_info`).
2. Use the target image as the output grid (shape + WCS define output).
3. Reproject source onto target grid using:
   - Flux-conserving method for `Jy/pixel`.
   - Interpolation method for `Jy/beam`.
4. Validate scientific invariants:
   - `Jy/pixel`: integrated flux conservation.
   - `Jy/beam`: peak and centroid stability, beam metadata consistency.

## Notes
- `xarray.interp` and `xesmf` are not WCS-aware; they do not handle sky-frame rotation or projection
  changes by themselves.
- For ICRS -> Galactic or rotated TAN projections, use a WCS reprojection tool rather than
  numeric coordinate interpolation.
