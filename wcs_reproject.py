"""
WCS-aware reprojection helpers for XRADIO image data.

Two user-facing entry points:
- reproject_to_match: reproject a source image onto the grid/WCS of a target image.
- reproject_to_frame: reproject a source image to a new sky frame, with an option
  to keep the existing pixel grid or re-center a same-sized grid in the new frame.

These functions operate on xarray Datasets or DataArrays and use astropy + reproject
under the hood. Reprojection is parallelized over non-spatial planes with dask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import xarray as xr

try:  # pragma: no cover - import guard for optional dependency
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from reproject import reproject_adaptive, reproject_exact, reproject_interp
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    _IMPORT_ERROR = exc
else:  # pragma: no cover - import guard for optional dependency
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class _WCSBuildResult:
    wcs: "WCS"
    coord_a: np.ndarray
    coord_b: np.ndarray


def reproject_to_match(
    source: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    *,
    data_var: str = "SKY",
    data_group: str | None = "base",
    dim_a: str = "l",
    dim_b: str = "m",
    method: str = "interp",
    order: int = 1,
) -> xr.Dataset | xr.DataArray:
    """
    Reproject `source` onto the WCS + grid of `target`.

    Parameters
    ----------
    source, target
        XRADIO Dataset or DataArray. For Dataset inputs, `data_var` selects
        the data variable to reproject.
    data_var
        Data variable name for Dataset inputs when `data_group` is not used.
    data_group
        Data group name for Dataset inputs (default: "base"). If available in
        `source.attrs["data_groups"]`, all group variables that include both
        spatial dims are reprojected. Set to None to force single-variable mode.
    dim_a, dim_b
        Spatial dimensions (default: l/m).
    method
        Reprojection algorithm: "interp", "exact", or "adaptive".
    order
        Interpolation order for "interp" (0=nearest, 1=bilinear, 3=cubic).

    Returns
    -------
    Dataset or DataArray
        Reprojected image with target spatial grid.
    Notes
    -----
    Spatial dimensions are treated as core dimensions for reprojection. If they
    are chunked, dask will rechunk them into single blocks.
    """
    _require_optional_deps()

    if isinstance(source, xr.Dataset):
        group_vars = _get_group_spatial_vars(
            source, data_group=data_group, dim_a=dim_a, dim_b=dim_b
        )
        if group_vars is not None:
            return _reproject_dataset_group_to_match(
                source,
                target,
                data_vars=group_vars,
                dim_a=dim_a,
                dim_b=dim_b,
                method=method,
                order=order,
            )

    src = _get_dataarray(source, data_var)
    tgt = _get_target_grid_dataarray(
        target, data_var=data_var, dim_a=dim_a, dim_b=dim_b
    )

    src_wcs = build_wcs_from_xradio(src, dim_a=dim_a, dim_b=dim_b)
    tgt_wcs = build_wcs_from_xradio(tgt, dim_a=dim_a, dim_b=dim_b)

    out = _reproject_dataarray(
        src,
        src_wcs.wcs,
        tgt_wcs.wcs,
        tgt_wcs.coord_a,
        tgt_wcs.coord_b,
        dim_a=dim_a,
        dim_b=dim_b,
        method=method,
        order=order,
    )

    return _attach_metadata(source, target, out, data_var=data_var)


def reproject_to_frame(
    source: xr.Dataset | xr.DataArray,
    frame: str,
    *,
    data_var: str = "SKY",
    data_group: str | None = "base",
    dim_a: str = "l",
    dim_b: str = "m",
    method: str = "interp",
    order: int = 1,
    keep_grid: bool = False,
    include_original_world_coords: bool = False,
) -> xr.Dataset | xr.DataArray:
    """
    Reproject `source` to a new sky frame.

    Parameters
    ----------
    source
        XRADIO Dataset or DataArray. For Dataset inputs, `data_var` selects
        the data variable to reproject.
    frame
        Target sky frame name.
        Supported strings (case-independent) are:
        - `"icrs"`
        - `"fk5"`
        - `"fk4"`
        - `"fk4noeterms"`
        - `"galactic"` (alias: `"gal"`)
        - `"geocentrictrueecliptic"`
        - `"geocentricmeanecliptic"`
        - `"barycentrictrueecliptic"`
        - `"barycentricmeanecliptic"`
    data_var
        Data variable name for Dataset inputs.
    data_group
        Data group name for Dataset inputs (default: `"base"`). If available in
        `source.attrs["data_groups"]`, all group variables that include both
        spatial dims are reprojected. If group metadata is missing or cannot be
        resolved to reprojectable spatial variables, fallback is single-variable
        mode using `data_var`. Set to `None` to force single-variable mode.
    dim_a, dim_b
        Spatial dimensions (default: l/m).
    method
        Reprojection algorithm: "interp", "exact", or "adaptive".
    order
        Interpolation order for "interp" (0=nearest, 1=bilinear, 3=cubic).
    keep_grid
        Controls whether the output uses the input pixel-coordinate lattice or a
        newly centered lattice:
        - If True, reuse the existing pixel grid (coords for `dim_a`/`dim_b`).
          The transformed output reference direction is still written to metadata
          and WCS, and remains tied to the `dim_a=0`, `dim_b=0` coordinate
          location defined by the reused grid.
        - If False, build a same-sized grid with the same pixel size but centered
          in offset coordinates, so `dim_a=0` and `dim_b=0` lie at the image
          midpoint (exact center pixel for odd sizes, midpoint between central
          pixels for even sizes). In this mode, the transformed output reference
          direction is associated with that centered `0,0` location.
    include_original_world_coords
        If True, include additional world-coordinate arrays for the original
        source frame on the output grid. These arrays are computed by
        transforming every output-pixel world coordinate from `frame` back into
        the source reference frame, so the mapping between target-frame and
        original-frame coordinates is one-to-one and pixel-wise consistent.
        Added coordinates use `original_` prefixes (for example
        `original_right_ascension`, `original_declination`).
    Returns
    -------
    Dataset or DataArray
        Reprojected image with the requested frame.
    Notes
    -----
    Spatial dimensions are treated as core dimensions for reprojection. If they
    are chunked, dask will rechunk them into single blocks.
    """
    _require_optional_deps()
    frame = frame.lower()

    if isinstance(source, xr.Dataset):
        try:
            group_vars = _get_group_spatial_vars(
                source, data_group=data_group, dim_a=dim_a, dim_b=dim_b
            )
        except (KeyError, ValueError):
            group_vars = None
        if group_vars is not None:
            return _reproject_dataset_group_to_frame(
                source,
                frame=frame,
                data_vars=group_vars,
                dim_a=dim_a,
                dim_b=dim_b,
                method=method,
                order=order,
                keep_grid=keep_grid,
                include_original_world_coords=include_original_world_coords,
            )

    src = _get_dataarray(source, data_var)
    src_wcs = build_wcs_from_xradio(src, dim_a=dim_a, dim_b=dim_b)

    ref_dir = _transform_reference_direction(src, frame)

    if keep_grid:
        tgt_wcs = build_wcs_from_xradio(
            src,
            dim_a=dim_a,
            dim_b=dim_b,
            frame_override=frame,
            ref_dir_override=ref_dir,
        )
    else:
        # Same shape and pixel size, but re-center the reference direction in
        # the target frame.
        coord_a, coord_b = _coords_for_same_pixel_size(src, dim_a=dim_a, dim_b=dim_b)
        tgt_wcs = build_wcs_from_xradio(
            src,
            dim_a=dim_a,
            dim_b=dim_b,
            frame_override=frame,
            ref_dir_override=ref_dir,
            coord_a=coord_a,
            coord_b=coord_b,
        )

    out = _reproject_dataarray(
        src,
        src_wcs.wcs,
        tgt_wcs.wcs,
        tgt_wcs.coord_a,
        tgt_wcs.coord_b,
        dim_a=dim_a,
        dim_b=dim_b,
        method=method,
        order=order,
    )

    result = _attach_metadata(
        source,
        None,
        out,
        data_var=data_var,
        frame_override=frame,
        ref_dir_override=ref_dir,
    )
    result = _replace_world_coords(
        result,
        wcs=tgt_wcs.wcs,
        dim_a=dim_a,
        dim_b=dim_b,
        frame=frame,
    )
    if include_original_world_coords:
        src_frame = _reference_frame_from_xradio(src)
        result = _add_original_world_coords(
            result,
            dim_a=dim_a,
            dim_b=dim_b,
            from_frame=frame,
            to_frame=src_frame,
        )
    if isinstance(result, xr.Dataset):
        result = _rotate_beam_pas_for_frame_change(source, result, frame=frame)
    return result


def _require_optional_deps() -> None:
    """Raise a clear runtime error when optional WCS dependencies are unavailable.

    Returns
    -------
    None
        This helper returns normally only when `astropy` and `reproject` imports
        succeeded at module import time.

    Raises
    ------
    RuntimeError
        Raised when optional dependencies failed to import. The original import
        exception text is included in the message for easier diagnosis.
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "astropy + reproject are required for WCS reprojection. "
            f"Import error: {_IMPORT_ERROR}"
        )


def _get_dataarray(obj: xr.Dataset | xr.DataArray, data_var: str) -> xr.DataArray:
    """Resolve an input object into a single `xarray.DataArray`.

    Parameters
    ----------
    obj
        Source object to extract from. If `obj` is a Dataset, `data_var` is used
        as the lookup key in `obj.data_vars`. If `obj` is already a DataArray, it
        is returned unchanged.
    data_var
        Data variable name used when `obj` is a Dataset.

    Returns
    -------
    xr.DataArray
        The selected data array that will be reprojected.

    Raises
    ------
    KeyError
        If `obj` is a Dataset and `data_var` is not present.
    """
    if isinstance(obj, xr.Dataset):
        if data_var not in obj.data_vars:
            raise KeyError(
                f"Dataset missing data_var={data_var!r}. "
                f"Available: {list(obj.data_vars)}"
            )
        return obj[data_var]
    return obj


def _flatten_group_var_names(value) -> list[str]:
    """Flatten nested XRADIO data-group definitions into a variable-name list.

    Parameters
    ----------
    value
        Group-definition value from `attrs["data_groups"][group_name]`. Supported
        shapes include:
        - `str`: single variable name,
        - `list`/`tuple`: nested collections of names and/or collections,
        - `dict`: nested mapping whose values contain the same structures.

    Returns
    -------
    list[str]
        Flat list of variable names. Unknown element types are ignored.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        names: list[str] = []
        for item in value.values():
            names.extend(_flatten_group_var_names(item))
        return names
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            names.extend(_flatten_group_var_names(item))
        return names
    return []


def _pa_basis_rotation_rad(
    *,
    ref_lon_rad: float,
    ref_lat_rad: float,
    src_frame: str,
    tgt_frame: str,
) -> float:
    """Compute local frame-basis rotation used for beam PA updates.

    Parameters
    ----------
    ref_lon_rad
        Reference longitude of the tangent point in radians, interpreted in
        `src_frame`.
    ref_lat_rad
        Reference latitude of the tangent point in radians, interpreted in
        `src_frame`.
    src_frame
        Source frame name understood by `astropy.coordinates.SkyCoord` (for
        example `"icrs"`, `"fk5"`, `"galactic"`).
    tgt_frame
        Target frame name understood by `astropy.coordinates.SkyCoord`.

    Returns
    -------
    float
        Position-angle rotation in radians, measured at the transformed tangent
        point from target-frame north toward the transformed source-frame north.

    Notes
    -----
    The implementation perturbs latitude by a tiny epsilon in `src_frame` to
    define a local north direction, transforms both points to `tgt_frame`, then
    computes the target-frame position angle between them.
    """
    # Small angular step to define local north direction in source frame.
    eps = 1e-7
    lat2 = np.clip(ref_lat_rad + eps, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
    center_src = SkyCoord(ref_lon_rad * u.rad, ref_lat_rad * u.rad, frame=src_frame)
    north_src = SkyCoord(ref_lon_rad * u.rad, lat2 * u.rad, frame=src_frame)
    center_tgt = center_src.transform_to(tgt_frame)
    north_tgt = north_src.transform_to(tgt_frame)
    return float(center_tgt.position_angle(north_tgt).to_value(u.rad))


def _rotate_beam_pas_for_frame_change(
    source: xr.Dataset | xr.DataArray, out: xr.Dataset, *, frame: str
) -> xr.Dataset:
    """Rotate `BEAM_FIT_PARAMS_*` PA values after a frame conversion.

    Parameters
    ----------
    source
        Original input object before reprojection. Beam PA updates are only
        attempted when this is a Dataset containing frame metadata in
        `attrs["coordinate_system_info"]`.
    out
        Output Dataset after reprojection. Any `BEAM_FIT_PARAMS_*` data variable
        with a `"beam_params_label"` coordinate and a `"pa"` label is updated.
    frame
        Target frame name (for example `"icrs"`, `"fk5"`, `"galactic"`).

    Returns
    -------
    xr.Dataset
        Dataset with updated beam position-angle values when applicable; unchanged
        otherwise.

    Notes
    -----
    PA values are updated with the opposite sign of the local basis rotation used
    in this package's world-coordinate convention. Variables missing expected beam
    metadata are skipped safely.
    """
    if not isinstance(source, xr.Dataset):
        return out

    src_csi = source.attrs.get("coordinate_system_info", {})
    src_ref = src_csi.get("reference_direction", {})
    src_attrs = src_ref.get("attrs", {})
    src_frame = src_attrs.get("frame", "icrs")
    src_vals = src_ref.get("data", [0.0, 0.0])
    if src_frame.lower() == frame.lower():
        return out

    rotation = _pa_basis_rotation_rad(
        ref_lon_rad=float(src_vals[0]),
        ref_lat_rad=float(src_vals[1]),
        src_frame=src_frame,
        tgt_frame=frame,
    )

    for name in list(out.data_vars):
        if not name.startswith("BEAM_FIT_PARAMS_"):
            continue
        beam = out[name]
        if (
            "beam_params_label" not in beam.dims
            or "beam_params_label" not in beam.coords
        ):
            continue
        labels = beam.coords["beam_params_label"].values
        pa_idx = np.where(labels == "pa")[0]
        if pa_idx.size == 0:
            continue

        pa_sel = int(pa_idx[0])
        updated = beam.copy(deep=True)
        updated_vals = np.asarray(updated.values)
        # Convert frame-basis rotation into beam-PA update (north->east convention).
        # Empirically for FK5->Galactic in this pipeline, PA follows the opposite
        # sign of the local basis angle returned above.
        updated_vals[..., pa_sel] = updated_vals[..., pa_sel] - rotation
        updated.values = updated_vals
        out[name] = updated

    return out


def _world_coord_names_for_frame(frame: str) -> tuple[str, str]:
    """Map a frame name to canonical world-coordinate variable names.

    Parameters
    ----------
    frame
        Frame selector. Supported families:
        - Galactic: `"galactic"`, `"gal"` -> Galactic coordinate names.
        - Ecliptic (easy, non-observer dependent): geocentric/barycentric true or
          mean ecliptic frame names -> Ecliptic names.
        - All other values map to equatorial names.

    Returns
    -------
    tuple[str, str]
        `(longitude_name, latitude_name)` coordinate names for the requested
        frame.
    """
    frame_lower = frame.lower()
    family = _frame_family(frame_lower)
    if family == "galactic":
        return "galactic_longitude", "galactic_latitude"
    if family == "ecliptic":
        return "ecliptic_longitude", "ecliptic_latitude"
    return "right_ascension", "declination"


def _frame_family(frame: str) -> str:
    """Return a normalized celestial frame family used for naming/CTYPE mapping.

    Parameters
    ----------
    frame
        Frame selector string, usually from API input or metadata.

    Returns
    -------
    str
        One of:
        - `"galactic"`
        - `"ecliptic"` (geocentric/barycentric true/mean variants)
        - `"equatorial"` (default fallback)
    """
    frame_lower = frame.lower()
    if frame_lower in {"galactic", "gal"}:
        return "galactic"
    if frame_lower in {
        "geocentrictrueecliptic",
        "geocentricmeanecliptic",
        "barycentrictrueecliptic",
        "barycentricmeanecliptic",
    }:
        return "ecliptic"
    return "equatorial"


def _reference_frame_from_xradio(obj: xr.DataArray | xr.Dataset) -> str:
    """Return the source reference-direction frame from XRADIO metadata.

    Parameters
    ----------
    obj
        Data object with optional `attrs["coordinate_system_info"]` metadata.

    Returns
    -------
    str
        Source frame name from metadata, defaulting to `"icrs"` when missing.
    """
    csi = obj.attrs.get("coordinate_system_info", {})
    ref_dir = csi.get("reference_direction", {})
    ref_attrs = ref_dir.get("attrs", {})
    return str(ref_attrs.get("frame", "icrs"))


def _known_world_coord_names() -> tuple[str, ...]:
    """Return canonical 2D world-coordinate names supported by this package.

    Returns
    -------
    tuple[str, ...]
        Canonical world-coordinate names used in Dataset/DataArray coordinates.
    """
    return (
        "right_ascension",
        "declination",
        "galactic_longitude",
        "galactic_latitude",
        "ecliptic_longitude",
        "ecliptic_latitude",
    )


def _known_world_coord_alias_names() -> tuple[str, ...]:
    """Return legacy world-coordinate alias names that should be removed.

    Returns
    -------
    tuple[str, ...]
        Alias names derived from canonical world-coordinate names in two forms:
        `input_<name>` and `<name>_input`.
    """
    aliases: list[str] = []
    for name in _known_world_coord_names():
        aliases.append(f"input_{name}")
        aliases.append(f"{name}_input")
    return tuple(aliases)


def _compute_world_coords_from_wcs(
    wcs: WCS, *, n_a: int, n_b: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate 2D longitude/latitude coordinates from a WCS on a pixel grid.

    Parameters
    ----------
    wcs
        Celestial WCS used to convert pixel indices to world coordinates.
    n_a
        Number of pixels along the first spatial axis.
    n_b
        Number of pixels along the second spatial axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays `(lon_rad, lat_rad)` with shape `(n_a, n_b)` in radians.
    """
    idx_a, idx_b = np.indices((n_a, n_b))
    lon_deg, lat_deg = wcs.wcs_pix2world(idx_a, idx_b, 0)
    return np.deg2rad(lon_deg), np.deg2rad(lat_deg)


def _transform_world_coords_between_frames(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    *,
    from_frame: str,
    to_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform 2D world-coordinate arrays between celestial frames.

    Parameters
    ----------
    lon_rad
        Longitude array in radians.
    lat_rad
        Latitude array in radians.
    from_frame
        Input frame name understood by `SkyCoord`.
    to_frame
        Output frame name understood by `SkyCoord`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed `(lon_rad, lat_rad)` arrays in radians with the same shape as
        inputs.
    """
    flat_lon = np.asarray(lon_rad).ravel()
    flat_lat = np.asarray(lat_rad).ravel()
    coord = SkyCoord(flat_lon * u.rad, flat_lat * u.rad, frame=from_frame)
    transformed = coord.transform_to(to_frame)
    out_lon = transformed.spherical.lon.to_value(u.rad).reshape(lon_rad.shape)
    out_lat = transformed.spherical.lat.to_value(u.rad).reshape(lat_rad.shape)
    return np.asarray(out_lon, dtype=float), np.asarray(out_lat, dtype=float)


def _replace_world_coords(
    obj: xr.Dataset | xr.DataArray,
    *,
    wcs: WCS,
    dim_a: str,
    dim_b: str,
    frame: str,
) -> xr.Dataset | xr.DataArray:
    """Replace world-coordinate axes on an object using a provided output WCS.

    Parameters
    ----------
    obj
        Dataset or DataArray whose coordinates are being updated.
    wcs
        Output celestial WCS used to compute replacement world-coordinate arrays.
    dim_a
        First spatial pixel dimension name.
    dim_b
        Second spatial pixel dimension name.
    frame
        Target frame selector controlling output coordinate names. Supported
        choices include `"galactic"`/`"gal"` for Galactic names and all other
        values for equatorial names.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Object with canonical world-coordinate arrays recomputed from `wcs`.

    Notes
    -----
    Existing canonical coordinates and known aliases are removed before assigning
    fresh frame-consistent coordinates.
    """
    lon_name, lat_name = _world_coord_names_for_frame(frame)
    existing_world = [name for name in _known_world_coord_names() if name in obj.coords]
    existing_aliases = [
        name for name in _known_world_coord_alias_names() if name in obj.coords
    ]

    if existing_world:
        obj = obj.drop_vars(existing_world)
    if existing_aliases:
        obj = obj.drop_vars(existing_aliases)

    n_a = obj.sizes[dim_a]
    n_b = obj.sizes[dim_b]
    lon_rad, lat_rad = _compute_world_coords_from_wcs(wcs, n_a=n_a, n_b=n_b)
    obj = obj.assign_coords(
        {
            lon_name: ((dim_a, dim_b), lon_rad),
            lat_name: ((dim_a, dim_b), lat_rad),
        }
    )
    return obj


def _add_original_world_coords(
    obj: xr.Dataset | xr.DataArray,
    *,
    dim_a: str,
    dim_b: str,
    from_frame: str,
    to_frame: str,
) -> xr.Dataset | xr.DataArray:
    """Add output-grid coordinates transformed from target frame to source frame.

    Parameters
    ----------
    obj
        Output data object that already contains canonical world coordinates for
        `from_frame`.
    dim_a
        First spatial dimension name.
    dim_b
        Second spatial dimension name.
    from_frame
        Frame currently represented by canonical world-coordinate arrays on
        `obj`.
    to_frame
        Original/source frame to write under `original_` coordinate names.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Object with additional coordinates:
        - `original_<lon_name>`
        - `original_<lat_name>`
        where base names are selected from `to_frame`.
    """
    from_lon_name, from_lat_name = _world_coord_names_for_frame(from_frame)
    to_lon_name, to_lat_name = _world_coord_names_for_frame(to_frame)

    lon_in = np.asarray(obj.coords[from_lon_name].values)
    lat_in = np.asarray(obj.coords[from_lat_name].values)
    lon_out, lat_out = _transform_world_coords_between_frames(
        lon_in,
        lat_in,
        from_frame=from_frame,
        to_frame=to_frame,
    )

    original_lon_name = f"original_{to_lon_name}"
    original_lat_name = f"original_{to_lat_name}"

    stale_originals = [
        name for name in (original_lon_name, original_lat_name) if name in obj.coords
    ]
    if stale_originals:
        obj = obj.drop_vars(stale_originals)

    obj = obj.assign_coords(
        {
            original_lon_name: ((dim_a, dim_b), lon_out),
            original_lat_name: ((dim_a, dim_b), lat_out),
        }
    )
    return obj


def _resolve_data_group_vars(ds: xr.Dataset, data_group: str) -> list[str]:
    """Resolve and validate variable names declared for a Dataset data group.

    Parameters
    ----------
    ds
        Dataset containing `attrs["data_groups"]`.
    data_group
        Group name key in `ds.attrs["data_groups"]` (commonly `"base"`).

    Returns
    -------
    list[str]
        De-duplicated list of data-variable names that exist in `ds`.

    Raises
    ------
    KeyError
        If `data_groups` metadata is missing, `data_group` is unknown, or no
        resolved names are present in Dataset variables.
    """
    groups = ds.attrs.get("data_groups")
    if not isinstance(groups, dict) or data_group not in groups:
        raise KeyError(
            f"Dataset missing data_group={data_group!r} in attrs['data_groups']. "
            f"Available groups: {list(groups) if isinstance(groups, dict) else []}"
        )

    names = _flatten_group_var_names(groups[data_group])
    names = [name for name in names if name in ds.data_vars]
    if not names:
        raise KeyError(
            f"data_group={data_group!r} resolved to no Dataset variables. "
            f"Available data vars: {list(ds.data_vars)}"
        )

    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _get_group_spatial_vars(
    ds: xr.Dataset, *, data_group: str | None, dim_a: str, dim_b: str
) -> list[str] | None:
    """Return group variables that include both requested spatial dimensions.

    Parameters
    ----------
    ds
        Source Dataset.
    data_group
        Group name in `ds.attrs["data_groups"]` to resolve. `None` disables
        group-mode selection and returns `None`.
    dim_a
        First required spatial dimension.
    dim_b
        Second required spatial dimension.

    Returns
    -------
    list[str] | None
        List of group variable names containing both spatial dimensions, or
        `None` when group mode is disabled or the group metadata/key is absent.

    Raises
    ------
    ValueError
        If the group exists but none of its resolved variables carry both spatial
        dimensions.
    """
    if data_group is None:
        return None

    groups = ds.attrs.get("data_groups")
    if not isinstance(groups, dict) or data_group not in groups:
        return None

    group_vars = _resolve_data_group_vars(ds, data_group)
    spatial_vars = [
        var for var in group_vars if dim_a in ds[var].dims and dim_b in ds[var].dims
    ]
    if not spatial_vars:
        raise ValueError(
            f"data_group={data_group!r} has no variables containing "
            f"spatial dims {dim_a!r}/{dim_b!r}."
        )
    return spatial_vars


def _reproject_dataset_group_to_match(
    source: xr.Dataset,
    target: xr.Dataset | xr.DataArray,
    *,
    data_vars: list[str],
    dim_a: str,
    dim_b: str,
    method: str,
    order: int,
) -> xr.Dataset:
    """Reproject all selected variables in a source Dataset data group.

    Parameters
    ----------
    source
        Source Dataset containing variables listed in `data_vars`.
    target
        Target grid object. If a Dataset is provided, its coordinates/attrs are
        used as output context; if a DataArray is provided, a new Dataset is
        created for outputs.
    data_vars
        Variable names to reproject. Each variable must include `dim_a` and
        `dim_b`.
    dim_a
        First spatial pixel dimension.
    dim_b
        Second spatial pixel dimension.
    method
        Reprojection method. Supported choices: `"interp"`, `"exact"`,
        `"adaptive"`.
    order
        Interpolation order for `"interp"` method (`0`, `1`, `3`, etc.).

    Returns
    -------
    xr.Dataset
        Dataset containing reprojected variables from `data_vars`, with metadata
        copied from `target` when it is a Dataset, otherwise from `source`.
    """
    src_ref = source[data_vars[0]]
    tgt_ref = _get_target_grid_dataarray(
        target, data_var=data_vars[0], dim_a=dim_a, dim_b=dim_b
    )

    src_wcs = build_wcs_from_xradio(src_ref, dim_a=dim_a, dim_b=dim_b)
    tgt_wcs = build_wcs_from_xradio(tgt_ref, dim_a=dim_a, dim_b=dim_b)

    out_ds = target.copy(deep=False) if isinstance(target, xr.Dataset) else xr.Dataset()
    for var in data_vars:
        out_ds[var] = _reproject_dataarray(
            source[var],
            src_wcs.wcs,
            tgt_wcs.wcs,
            tgt_wcs.coord_a,
            tgt_wcs.coord_b,
            dim_a=dim_a,
            dim_b=dim_b,
            method=method,
            order=order,
        )

    if isinstance(target, xr.Dataset):
        out_ds.attrs = dict(target.attrs)
    else:
        out_ds.attrs = dict(source.attrs)

    # Preserve source group definitions for downstream selection logic.
    if "data_groups" in source.attrs:
        out_ds.attrs["data_groups"] = source.attrs["data_groups"]
    return out_ds


def _reproject_dataset_group_to_frame(
    source: xr.Dataset,
    *,
    frame: str,
    data_vars: list[str],
    dim_a: str,
    dim_b: str,
    method: str,
    order: int,
    keep_grid: bool,
    include_original_world_coords: bool,
) -> xr.Dataset:
    """Reproject all selected group variables in a Dataset to a target frame.

    Parameters
    ----------
    source
        Source Dataset containing variables listed in `data_vars`.
    frame
        Target sky frame name, case-normalized by the caller.
    data_vars
        Variable names to reproject. Each variable must include both `dim_a` and
        `dim_b`.
    dim_a
        First spatial pixel dimension.
    dim_b
        Second spatial pixel dimension.
    method
        Reprojection method. Supported choices: `"interp"`, `"exact"`,
        `"adaptive"`.
    order
        Interpolation order for `"interp"` method (`0`, `1`, `3`, etc.).
    keep_grid
        If `True`, keep input spatial coordinate arrays. If `False`, rebuild a
        same-sized centered offset grid with the same pixel spacing.
    include_original_world_coords
        If `True`, add transformed `original_*` coordinate arrays on output
        pixels for the source frame.

    Returns
    -------
    xr.Dataset
        Dataset with all selected group variables reprojected to `frame`,
        frame-consistent world coordinates, updated frame metadata, and beam-PA
        updates when beam metadata variables are present.
    """
    src_ref = source[data_vars[0]]
    src_wcs = build_wcs_from_xradio(src_ref, dim_a=dim_a, dim_b=dim_b)
    ref_dir = _transform_reference_direction(src_ref, frame)

    if keep_grid:
        tgt_wcs = build_wcs_from_xradio(
            src_ref,
            dim_a=dim_a,
            dim_b=dim_b,
            frame_override=frame,
            ref_dir_override=ref_dir,
        )
    else:
        coord_a, coord_b = _coords_for_same_pixel_size(
            src_ref, dim_a=dim_a, dim_b=dim_b
        )
        tgt_wcs = build_wcs_from_xradio(
            src_ref,
            dim_a=dim_a,
            dim_b=dim_b,
            frame_override=frame,
            ref_dir_override=ref_dir,
            coord_a=coord_a,
            coord_b=coord_b,
        )

    out_ds = source.copy(deep=False)
    dim_coord_updates = {
        dim_a: (dim_a, tgt_wcs.coord_a),
        dim_b: (dim_b, tgt_wcs.coord_b),
    }
    out_ds = out_ds.assign_coords(dim_coord_updates)

    stale_world_aliases = [
        name for name in _known_world_coord_alias_names() if name in out_ds.coords
    ]
    if stale_world_aliases:
        out_ds = out_ds.drop_vars(stale_world_aliases)

    for var in data_vars:
        out_var = _reproject_dataarray(
            source[var],
            src_wcs.wcs,
            tgt_wcs.wcs,
            tgt_wcs.coord_a,
            tgt_wcs.coord_b,
            dim_a=dim_a,
            dim_b=dim_b,
            method=method,
            order=order,
        )
        _update_frame_attrs(out_var, frame, ref_dir_override=ref_dir)
        out_ds[var] = out_var

    out_ds.attrs = dict(source.attrs)
    if "coordinate_system_info" in out_ds.attrs:
        out_ds.attrs["coordinate_system_info"] = _update_csi_frame(
            out_ds.attrs["coordinate_system_info"],
            frame,
            source[data_vars[0]],
            ref_dir_override=ref_dir,
        )

    out_ds = _replace_world_coords(
        out_ds,
        wcs=tgt_wcs.wcs,
        dim_a=dim_a,
        dim_b=dim_b,
        frame=frame,
    )
    if include_original_world_coords:
        src_frame = _reference_frame_from_xradio(src_ref)
        out_ds = _add_original_world_coords(
            out_ds,
            dim_a=dim_a,
            dim_b=dim_b,
            from_frame=frame,
            to_frame=src_frame,
        )
    out_ds = _rotate_beam_pas_for_frame_change(source, out_ds, frame=frame)
    return out_ds


def _get_target_grid_dataarray(
    obj: xr.Dataset | xr.DataArray,
    *,
    data_var: str,
    dim_a: str,
    dim_b: str,
) -> xr.DataArray:
    """Return a DataArray that defines the target reprojection grid and WCS.

    Parameters
    ----------
    obj
        Target object provided by the caller.
    data_var
        Preferred data-variable name when `obj` is a Dataset.
    dim_a
        First spatial dimension name required when synthesizing a grid-only
        DataArray from Dataset coordinates.
    dim_b
        Second spatial dimension name required when synthesizing a grid-only
        DataArray from Dataset coordinates.

    Returns
    -------
    xr.DataArray
        Existing target DataArray, requested Dataset variable, or a synthetic
        zero-valued DataArray carrying target spatial coordinates and optional
        `coordinate_system_info`.

    Raises
    ------
    KeyError
        If `obj` is a Dataset lacking both `data_var` and required spatial
        coordinates `dim_a`/`dim_b`.
    """
    if isinstance(obj, xr.DataArray):
        return obj

    if data_var in obj.data_vars:
        return obj[data_var]

    # Allow a target Dataset that only carries grid coordinates + WCS metadata.
    if dim_a not in obj.coords or dim_b not in obj.coords:
        raise KeyError(
            f"Dataset missing data_var={data_var!r} and required coords "
            f"{dim_a!r}/{dim_b!r}. Available data vars: {list(obj.data_vars)}; "
            f"coords: {list(obj.coords)}"
        )

    coord_a = np.asarray(obj[dim_a].values)
    coord_b = np.asarray(obj[dim_b].values)
    da = xr.DataArray(
        np.zeros((coord_a.size, coord_b.size), dtype=np.float64),
        dims=(dim_a, dim_b),
        coords={dim_a: coord_a, dim_b: coord_b},
    )
    if "coordinate_system_info" in obj.attrs:
        da.attrs["coordinate_system_info"] = obj.attrs["coordinate_system_info"]
    return da


def _attach_metadata(
    source: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray | None,
    out: xr.DataArray,
    *,
    data_var: str,
    frame_override: str | None = None,
    ref_dir_override: Iterable[float] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Attach attrs/coordinates around a reprojected DataArray result.

    Parameters
    ----------
    source
        Original input object used for reprojection.
    target
        Optional target object. When present and a Dataset, target attrs/context
        are preferred for match-mode outputs.
    out
        Reprojected DataArray to attach into a DataArray or Dataset container.
    data_var
        Data variable name to assign when returning a Dataset.
    frame_override
        Optional frame name applied to output `coordinate_system_info` in
        frame-conversion workflows.
    ref_dir_override
        Optional reference direction `[lon, lat]` in radians written into updated
        `coordinate_system_info` when `frame_override` is provided.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Output object with reprojected data inserted and metadata preserved or
        updated for the selected workflow.

    Notes
    -----
    Known world-coordinate alias coordinates (`input_*` and `*_input`) are
    dropped from Dataset outputs to avoid stale coordinate confusion.
    """
    if isinstance(source, xr.DataArray):
        out.attrs = dict(source.attrs)
        if frame_override:
            _update_frame_attrs(out, frame_override, ref_dir_override=ref_dir_override)
        return out

    if target is not None and isinstance(target, xr.Dataset):
        ds = target.copy(deep=False)
        if data_var in ds.data_vars:
            ds = ds.drop_vars([data_var])
    else:
        ds = source.copy(deep=False)
        ds = ds.drop_vars([data_var])
        # When output grid changes (for example keep_grid=False in frame reprojection),
        # replace source core-dimension coords before assignment to avoid
        # reindex-to-source alignment that can fill output data with NaNs.
        dim_coord_updates = {
            dim: out.coords[dim]
            for dim in out.dims
            if dim in out.coords and dim in ds.dims
        }
        if dim_coord_updates:
            ds = ds.assign_coords(dim_coord_updates)
    stale_world_aliases = [
        name for name in _known_world_coord_alias_names() if name in ds.coords
    ]
    if stale_world_aliases:
        ds = ds.drop_vars(stale_world_aliases)
    ds[data_var] = out
    if target is not None and isinstance(target, xr.Dataset):
        ds.attrs = dict(target.attrs)
    else:
        ds.attrs = dict(source.attrs)
        if frame_override:
            _update_frame_attrs(
                ds[data_var], frame_override, ref_dir_override=ref_dir_override
            )
            if "coordinate_system_info" in ds.attrs:
                ds.attrs["coordinate_system_info"] = _update_csi_frame(
                    ds.attrs["coordinate_system_info"],
                    frame_override,
                    out,
                    ref_dir_override=ref_dir_override,
                )
    return ds


def _update_frame_attrs(
    da: xr.DataArray, frame: str, *, ref_dir_override: Iterable[float] | None = None
) -> None:
    """Update frame metadata in a DataArray's `coordinate_system_info` attribute.

    Parameters
    ----------
    da
        DataArray whose `attrs["coordinate_system_info"]` may be updated.
    frame
        Target frame name to write into reference-direction attrs.
    ref_dir_override
        Optional replacement reference direction `[lon, lat]` in radians.

    Returns
    -------
    None
        The DataArray is updated in place when metadata is present.
    """
    if "coordinate_system_info" not in da.attrs:
        return
    da.attrs["coordinate_system_info"] = _update_csi_frame(
        da.attrs["coordinate_system_info"],
        frame,
        da,
        ref_dir_override=ref_dir_override,
    )


def _update_csi_frame(
    csi: dict,
    frame: str,
    da: xr.DataArray,
    *,
    ref_dir_override: Iterable[float] | None = None,
) -> dict:
    """Return a shallow-copied CSI dict with updated frame/reference direction.

    Parameters
    ----------
    csi
        Coordinate-system-info mapping in XRADIO style.
    frame
        Frame name written into
        `csi["reference_direction"]["attrs"]["frame"]` when that structure
        exists.
    da
        DataArray associated with this metadata update. Present for signature
        symmetry with callers; not used directly in the current implementation.
    ref_dir_override
        Optional replacement `[lon, lat]` in radians written to
        `csi["reference_direction"]["data"]`.

    Returns
    -------
    dict
        New dictionary containing updated reference-direction metadata when
        available; otherwise equivalent to the input mapping.
    """
    new_csi = dict(csi)
    if "reference_direction" in new_csi:
        rd = dict(new_csi["reference_direction"])
        attrs = dict(rd.get("attrs", {}))
        attrs["frame"] = frame
        rd["attrs"] = attrs
        if ref_dir_override is not None:
            rd["data"] = list(ref_dir_override)
        new_csi["reference_direction"] = rd
    return new_csi


def _coords_for_same_pixel_size(
    src: xr.DataArray,
    *,
    dim_a: str,
    dim_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a centered output grid preserving source pixel spacing.

    Parameters
    ----------
    src
        Source DataArray containing spatial coordinates for `dim_a` and `dim_b`.
    dim_a
        First spatial dimension used to estimate output pixel spacing.
    dim_b
        Second spatial dimension used to estimate output pixel spacing.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New `(coord_a, coord_b)` arrays with the same lengths and median pixel
        spacings as input axes, centered around zero.
    """
    coord_a_src = np.asarray(src[dim_a].values)
    coord_b_src = np.asarray(src[dim_b].values)
    cdelt_a = float(np.nanmedian(np.diff(coord_a_src)))
    cdelt_b = float(np.nanmedian(np.diff(coord_b_src)))
    n_a = coord_a_src.size
    n_b = coord_b_src.size

    center_a = (n_a - 1) / 2.0
    center_b = (n_b - 1) / 2.0
    coord_a = (np.arange(n_a) - center_a) * cdelt_a
    coord_b = (np.arange(n_b) - center_b) * cdelt_b
    return coord_a, coord_b


def build_wcs_from_xradio(
    da: xr.Dataset | xr.DataArray,
    *,
    dim_a: str,
    dim_b: str,
    frame_override: str | None = None,
    ref_dir_override: Iterable[float] | None = None,
    coord_a: np.ndarray | None = None,
    coord_b: np.ndarray | None = None,
) -> _WCSBuildResult:
    """
    Build a 2D celestial `astropy.wcs.WCS` from XRADIO-style metadata and axes.

    Parameters
    ----------
    da
        Input `xarray.DataArray` or `xarray.Dataset` containing
        `attrs["coordinate_system_info"]` in XRADIO style. This metadata is
        used to read projection, reference direction, optional frame, and
        optional pixel coordinate transformation (`pc`) matrix.
    dim_a
        Name of the first spatial pixel axis in `da` (for example `"l"`).
        The coordinate values are assumed to be angular offsets in radians and
        must be monotonic enough to infer a representative pixel spacing from
        `np.diff`.
    dim_b
        Name of the second spatial pixel axis in `da` (for example `"m"`).
        The coordinate values are assumed to be angular offsets in radians and
        must be monotonic enough to infer a representative pixel spacing from
        `np.diff`.
    frame_override
        Optional sky-frame override used for output celestial axis types.
        If `None`, the frame is read from
        `coordinate_system_info["reference_direction"]["attrs"]["frame"]`
        and defaults to `"icrs"` when missing.
        Supported strings are case-independent.
        Common values include `"icrs"`, `"fk5"`, `"fk4"`, `"fk4noeterms"`,
        `"galactic"` (or `"gal"`), and ecliptic names such as
        `"geocentrictrueecliptic"`:
        `"galactic"`/`"gal"` maps to `GLON/GLAT`;
        `geocentric*ecliptic`/`barycentric*ecliptic` map to `ELON/ELAT`;
        all other values map to `RA/DEC`.
    ref_dir_override
        Optional two-element iterable `[lon, lat]` in radians to use as WCS
        reference direction (`CRVAL`). If `None`, values are read from
        `coordinate_system_info["reference_direction"]["data"]`, defaulting to
        `[0.0, 0.0]` when missing.
    coord_a
        Optional explicit coordinates for `dim_a` in radians. When provided,
        these values are used instead of `da[dim_a]` to compute `CDELT1` and
        `CRPIX1`.
    coord_b
        Optional explicit coordinates for `dim_b` in radians. When provided,
        these values are used instead of `da[dim_b]` to compute `CDELT2` and
        `CRPIX2`.

    Returns
    -------
    _WCSBuildResult
        Dataclass bundle containing:
        - `wcs`: constructed 2D celestial `astropy.wcs.WCS`,
        - `coord_a`: effective `dim_a` coordinate array used for WCS inference,
        - `coord_b`: effective `dim_b` coordinate array used for WCS inference.

    Notes
    -----
    - Input coordinate arrays and reference-direction metadata are interpreted
      in radians; the produced WCS stores angular quantities in degrees as
      required by FITS-WCS conventions.
    - Pixel reference (`CRPIX`) is computed in 0-based index space and then
      shifted to FITS 1-based convention.
    - If present, `coordinate_system_info["pixel_coordinate_transformation_matrix"]`
      is copied directly into `wcs.wcs.pc`.
    - This helper assumes a 2D celestial WCS and does not encode spectral or
      Stokes axes.

    Raises
    ------
    KeyError
        If required coordinate variables `dim_a` or `dim_b` are missing from
        `da` and corresponding `coord_a`/`coord_b` overrides are not provided.
    ValueError
        If inferred coordinate spacing for either spatial axis is zero.
    IndexError
        If a coordinate array has fewer than two elements, spacing cannot be
        inferred.
    """
    csi = da.attrs.get("coordinate_system_info", {})
    projection = csi.get("projection", "SIN")

    ref_dir = csi.get("reference_direction", {})
    ref_attrs = ref_dir.get("attrs", {})
    frame = frame_override or ref_attrs.get("frame", "icrs")
    ref_vals = ref_dir_override or ref_dir.get("data", [0.0, 0.0])

    coord_a = np.asarray(da[dim_a].values if coord_a is None else coord_a)
    coord_b = np.asarray(da[dim_b].values if coord_b is None else coord_b)

    cdelt_a, crpix_a = _coord_to_cdelt_crpix(coord_a)
    cdelt_b, crpix_b = _coord_to_cdelt_crpix(coord_b)

    ctype1, ctype2 = _frame_to_ctype(frame, projection)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [ctype1, ctype2]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.cdelt = [np.rad2deg(cdelt_a), np.rad2deg(cdelt_b)]
    wcs.wcs.crpix = [crpix_a + 1.0, crpix_b + 1.0]
    wcs.wcs.crval = [np.rad2deg(ref_vals[0]), np.rad2deg(ref_vals[1])]
    # Store image shape on the high-level WCS object; Wcsprm does not expose
    # writable naxis1/naxis2 attributes.
    wcs.pixel_shape = (int(coord_a.size), int(coord_b.size))
    wcs.array_shape = (int(coord_b.size), int(coord_a.size))

    pc = csi.get("pixel_coordinate_transformation_matrix")
    if pc is not None:
        wcs.wcs.pc = np.asarray(pc, dtype=float)

    return _WCSBuildResult(wcs=wcs, coord_a=coord_a, coord_b=coord_b)


def _coord_to_cdelt_crpix(coord: np.ndarray) -> Tuple[float, float]:
    """Infer FITS-WCS axis spacing and reference-pixel index from coordinates.

    Parameters
    ----------
    coord
        One-dimensional spatial coordinate array in radians.

    Returns
    -------
    tuple[float, float]
        `(cdelt, crpix0)` where `cdelt` is median axis spacing and `crpix0` is
        the 0-based pixel index where coordinate value zero is interpolated.

    Raises
    ------
    ValueError
        If inferred spacing is exactly zero.
    IndexError
        If fewer than two coordinates are provided (via `coord[1]` access).
    """
    coord = np.asarray(coord)
    diffs = np.diff(coord)
    cdelt = float(np.nanmedian(diffs))
    if cdelt == 0.0:
        raise ValueError("Coordinate spacing is zero; cannot build WCS.")

    x = coord
    y = np.arange(coord.size, dtype=float)
    if x[1] < x[0]:
        x = x[::-1]
        y = y[::-1]
    crpix = float(np.interp(0.0, x, y))
    return cdelt, crpix


def _frame_to_ctype(frame: str, projection: str) -> Tuple[str, str]:
    """Convert frame/projection inputs into FITS `CTYPE1`/`CTYPE2` strings.

    Parameters
    ----------
    frame
        Sky frame selector. Supported families:
        - Galactic (`"galactic"`, `"gal"`) -> `GLON`/`GLAT`
        - Ecliptic (`geocentric*ecliptic`, `barycentric*ecliptic`) ->
          `ELON`/`ELAT`
        - All other values -> `RA`/`DEC`
    projection
        FITS projection suffix such as `"SIN"` or `"TAN"`.

    Returns
    -------
    tuple[str, str]
        `(ctype1, ctype2)` strings ready for assignment to `wcs.wcs.ctype`.
    """
    family = _frame_family(frame)
    if family == "galactic":
        return f"GLON-{projection}", f"GLAT-{projection}"
    if family == "ecliptic":
        return f"ELON-{projection}", f"ELAT-{projection}"
    return f"RA---{projection}", f"DEC--{projection}"


def _transform_reference_direction(da: xr.DataArray, frame: str) -> tuple[float, float]:
    """Transform a DataArray reference direction into a target sky frame.

    Parameters
    ----------
    da
        DataArray carrying XRADIO `coordinate_system_info` metadata.
    frame
        Target frame name understood by `SkyCoord.transform_to`.

    Returns
    -------
    tuple[float, float]
        `(lon_rad, lat_rad)` in radians for the transformed reference direction.

    Notes
    -----
    If source reference-direction metadata is missing, the transformation starts
    from `[0.0, 0.0]` in an `"icrs"` frame default.
    """
    csi = da.attrs.get("coordinate_system_info", {})
    ref_dir = csi.get("reference_direction", {})
    ref_attrs = ref_dir.get("attrs", {})
    ref_vals = ref_dir.get("data", [0.0, 0.0])
    ref_frame = ref_attrs.get("frame", "icrs")

    coord = SkyCoord(ref_vals[0] * u.rad, ref_vals[1] * u.rad, frame=ref_frame)
    coord_tgt = coord.transform_to(frame)
    return float(coord_tgt.spherical.lon.to_value(u.rad)), float(
        coord_tgt.spherical.lat.to_value(u.rad)
    )


def _reproject_dataarray(
    src: xr.DataArray,
    wcs_in: WCS,
    wcs_out: WCS,
    coord_a_out: np.ndarray,
    coord_b_out: np.ndarray,
    *,
    dim_a: str,
    dim_b: str,
    method: str,
    order: int,
) -> xr.DataArray:
    """Reproject a DataArray plane-by-plane across non-spatial dimensions.

    Parameters
    ----------
    src
        Input DataArray containing spatial dimensions `dim_a` and `dim_b`.
    wcs_in
        Input celestial WCS describing `src`.
    wcs_out
        Output celestial WCS describing the target grid.
    coord_a_out
        Output coordinate values for `dim_a`.
    coord_b_out
        Output coordinate values for `dim_b`.
    dim_a
        First spatial core dimension name in `src`.
    dim_b
        Second spatial core dimension name in `src`.
    method
        Reprojection method. Supported choices: `"interp"`, `"exact"`,
        `"adaptive"`.
    order
        Interpolation order used only for `"interp"` (`0`, `1`, `3`, etc.).

    Returns
    -------
    xr.DataArray
        Reprojected DataArray with the same non-spatial dimensions as `src` and
        output spatial coordinates `(coord_a_out, coord_b_out)`.

    Notes
    -----
    Reprojection kernels operate in `(y, x)` order, so each `(dim_a, dim_b)` plane
    is transposed before/after calling `reproject`.
    """
    reproject_func = _select_reproject_func(method)
    shape_out = (coord_b_out.size, coord_a_out.size)
    out_dim_a = f"{dim_a}__out"
    out_dim_b = f"{dim_b}__out"

    def _reproject_plane(plane: np.ndarray) -> np.ndarray:
        """Reproject a single 2D spatial plane in NumPy array form.

        Parameters
        ----------
        plane
            Plane with axis order `(dim_a, dim_b)`.

        Returns
        -------
        np.ndarray
            Reprojected plane with the same axis order `(dim_a, dim_b)`.
        """
        arr = np.asarray(plane)
        # Reproject expects array order (y, x). For our (dim_a, dim_b) core
        # dims, transpose to (dim_b, dim_a) for reprojection, then transpose
        # back so output keeps (dim_a, dim_b).
        arr_for_reproject = arr.T
        if method == "interp":
            out, _ = reproject_func(
                (arr_for_reproject, wcs_in), wcs_out, shape_out=shape_out, order=order
            )
        else:
            out, _ = reproject_func(
                (arr_for_reproject, wcs_in), wcs_out, shape_out=shape_out
            )
        return out.T

    out = xr.apply_ufunc(
        _reproject_plane,
        src,
        input_core_dims=[[dim_a, dim_b]],
        output_core_dims=[[out_dim_a, out_dim_b]],
        exclude_dims={dim_a, dim_b},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {out_dim_a: coord_a_out.size, out_dim_b: coord_b_out.size}
        },
    )

    out = out.rename({out_dim_a: dim_a, out_dim_b: dim_b})
    out = out.assign_coords({dim_a: coord_a_out, dim_b: coord_b_out})
    out.attrs = dict(src.attrs)
    return out


def _select_reproject_func(method: str):
    """Return the reproject backend function for a method selector.

    Parameters
    ----------
    method
        Method selector. Supported choices are:
        - `"interp"`: interpolation-based reprojection,
        - `"exact"`: flux-conserving footprint overlap,
        - `"adaptive"`: adaptive resampling.

    Returns
    -------
    callable
        Reproject function object from the `reproject` package.

    Raises
    ------
    ValueError
        If `method` is not one of the supported choices.
    """
    method_lower = method.lower()
    if method_lower == "interp":
        return reproject_interp
    if method_lower == "exact":
        return reproject_exact
    if method_lower == "adaptive":
        return reproject_adaptive
    raise ValueError(
        f"Unknown method={method!r}. Expected 'interp', 'exact', or 'adaptive'."
    )
