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

    src_wcs = _build_wcs_from_xradio(src, dim_a=dim_a, dim_b=dim_b)
    tgt_wcs = _build_wcs_from_xradio(tgt, dim_a=dim_a, dim_b=dim_b)

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
    dim_a: str = "l",
    dim_b: str = "m",
    method: str = "interp",
    order: int = 1,
    keep_grid: bool = False,
    update_world_coords: bool = True,
    keep_input_world_coords: bool = False,
) -> xr.Dataset | xr.DataArray:
    """
    Reproject `source` to a new sky frame.

    Parameters
    ----------
    source
        XRADIO Dataset or DataArray. For Dataset inputs, `data_var` selects
        the data variable to reproject.
    frame
        Target sky frame name (e.g. "icrs", "fk5", "galactic").
    data_var
        Data variable name for Dataset inputs.
    dim_a, dim_b
        Spatial dimensions (default: l/m).
    method
        Reprojection algorithm: "interp", "exact", or "adaptive".
    order
        Interpolation order for "interp" (0=nearest, 1=bilinear, 3=cubic).
    keep_grid
        If True, keep the existing pixel grid (coords for dim_a/dim_b). If False,
        build a same-sized grid with the same pixel size but centered in the
        target frame.
    update_world_coords
        If True, regenerate 2D world-coordinate axes from the output WCS so they
        match the requested frame, but only when input world-coordinate axes are
        present.
    keep_input_world_coords
        If True and `update_world_coords=True`, preserve original world-coordinate
        axes under `input_*` names (for example `input_right_ascension`) before
        writing frame-consistent output coordinates.

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

    src = _get_dataarray(source, data_var)
    input_has_world_coords = any(
        name in src.coords for name in _known_world_coord_names()
    )
    src_wcs = _build_wcs_from_xradio(src, dim_a=dim_a, dim_b=dim_b)
    src_frame, src_ref_vals = _get_source_frame_and_ref_dir(src)

    ref_dir = _transform_reference_direction(src, frame)

    if keep_grid:
        tgt_wcs = _build_wcs_from_xradio(
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
        tgt_wcs = _build_wcs_from_xradio(
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
    if update_world_coords and input_has_world_coords:
        result = _replace_world_coords(
            result,
            wcs=tgt_wcs.wcs,
            dim_a=dim_a,
            dim_b=dim_b,
            frame=frame,
            keep_input=keep_input_world_coords,
        )
    if isinstance(result, xr.Dataset):
        result = _rotate_beam_pas_for_frame_change(source, result, frame=frame)
    return result


def _require_optional_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "astropy + reproject are required for WCS reprojection. "
            f"Import error: {_IMPORT_ERROR}"
        )


def _get_dataarray(obj: xr.Dataset | xr.DataArray, data_var: str) -> xr.DataArray:
    if isinstance(obj, xr.Dataset):
        if data_var not in obj.data_vars:
            raise KeyError(
                f"Dataset missing data_var={data_var!r}. "
                f"Available: {list(obj.data_vars)}"
            )
        return obj[data_var]
    return obj


def _flatten_group_var_names(value) -> list[str]:
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


def _get_source_frame_and_ref_dir(da: xr.DataArray) -> tuple[str, list[float]]:
    csi = da.attrs.get("coordinate_system_info", {})
    ref_dir = csi.get("reference_direction", {})
    ref_attrs = ref_dir.get("attrs", {})
    frame = ref_attrs.get("frame", "icrs")
    ref_vals = ref_dir.get("data", [0.0, 0.0])
    return frame, ref_vals


def _pa_basis_rotation_rad(
    *,
    ref_lon_rad: float,
    ref_lat_rad: float,
    src_frame: str,
    tgt_frame: str,
) -> float:
    """Compute local PA rotation from source frame north to target-frame basis."""
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
    """Rotate beam position angles to match frame-changed image orientation."""
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
    frame_lower = frame.lower()
    if frame_lower in {"galactic", "gal"}:
        return "galactic_longitude", "galactic_latitude"
    return "right_ascension", "declination"


def _known_world_coord_names() -> tuple[str, ...]:
    return (
        "right_ascension",
        "declination",
        "galactic_longitude",
        "galactic_latitude",
    )


def _compute_world_coords_from_wcs(
    wcs: WCS, *, n_a: int, n_b: int
) -> tuple[np.ndarray, np.ndarray]:
    idx_a, idx_b = np.indices((n_a, n_b))
    lon_deg, lat_deg = wcs.wcs_pix2world(idx_a, idx_b, 0)
    return np.deg2rad(lon_deg), np.deg2rad(lat_deg)


def _replace_world_coords(
    obj: xr.Dataset | xr.DataArray,
    *,
    wcs: WCS,
    dim_a: str,
    dim_b: str,
    frame: str,
    keep_input: bool,
) -> xr.Dataset | xr.DataArray:
    lon_name, lat_name = _world_coord_names_for_frame(frame)
    existing_world = [name for name in _known_world_coord_names() if name in obj.coords]

    if keep_input:
        for name in existing_world:
            input_name = f"input_{name}"
            if input_name not in obj.coords:
                obj = obj.assign_coords({input_name: obj.coords[name]})
    if existing_world:
        obj = obj.drop_vars(existing_world)

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


def _resolve_data_group_vars(ds: xr.Dataset, data_group: str) -> list[str]:
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
    src_ref = source[data_vars[0]]
    tgt_ref = _get_target_grid_dataarray(
        target, data_var=data_vars[0], dim_a=dim_a, dim_b=dim_b
    )

    src_wcs = _build_wcs_from_xradio(src_ref, dim_a=dim_a, dim_b=dim_b)
    tgt_wcs = _build_wcs_from_xradio(tgt_ref, dim_a=dim_a, dim_b=dim_b)

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


def _get_target_grid_dataarray(
    obj: xr.Dataset | xr.DataArray,
    *,
    data_var: str,
    dim_a: str,
    dim_b: str,
) -> xr.DataArray:
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


def _build_wcs_from_xradio(
    da: xr.DataArray,
    *,
    dim_a: str,
    dim_b: str,
    frame_override: str | None = None,
    ref_dir_override: Iterable[float] | None = None,
    coord_a: np.ndarray | None = None,
    coord_b: np.ndarray | None = None,
) -> _WCSBuildResult:
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

    pc = csi.get("pixel_coordinate_transformation_matrix")
    if pc is not None:
        wcs.wcs.pc = np.asarray(pc, dtype=float)

    return _WCSBuildResult(wcs=wcs, coord_a=coord_a, coord_b=coord_b)


def _coord_to_cdelt_crpix(coord: np.ndarray) -> Tuple[float, float]:
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
    frame_lower = frame.lower()
    if frame_lower in {"galactic", "gal"}:
        return f"GLON-{projection}", f"GLAT-{projection}"
    return f"RA---{projection}", f"DEC--{projection}"


def _transform_reference_direction(da: xr.DataArray, frame: str) -> tuple[float, float]:
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
    reproject_func = _select_reproject_func(method)
    shape_out = (coord_b_out.size, coord_a_out.size)
    out_dim_a = f"{dim_a}__out"
    out_dim_b = f"{dim_b}__out"

    def _reproject_plane(plane: np.ndarray) -> np.ndarray:
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
