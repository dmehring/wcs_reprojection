"""Regression tests for frame reprojection to Galactic coordinates."""

from __future__ import annotations

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.coordinates import SkyCoord
from xradio.image import make_empty_sky_image

from wcs_reproject import reproject_to_frame


def _make_point_source_dataset(
    *,
    n_l: int = 128,
    n_m: int = 128,
    cell_arcsec: float = 2.0,
    loc: tuple[int, int] = (90, 40),
) -> xr.Dataset:
    """Create an off-center point-source image with FK5 world coordinates."""
    cell = np.deg2rad(cell_arcsec / 3600.0)
    xds = make_empty_sky_image(
        phase_center=[0.0, 0.0],
        image_size=[n_l, n_m],
        cell_size=[cell, cell],
        frequency_coords=np.array([1.4e9]),
        pol_coords=["I"],
        time_coords=np.array([59000.0]),
        direction_reference="fk5",
        projection="SIN",
        spectral_reference="lsrk",
        do_sky_coords=True,
    )
    base = np.zeros((n_l, n_m), dtype=np.float64)
    base[loc] = 1.0
    xds["SKY"] = xr.DataArray(
        base[None, None, None, :, :],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            d: xds.coords[d] for d in ("time", "frequency", "polarization", "l", "m")
        },
    )
    xds["SKY"].attrs["units"] = "Jy/pixel"
    return xds


def _add_legacy_input_world_aliases(xds: xr.Dataset) -> xr.Dataset:
    """Add legacy `*_input` world-coordinate aliases to mimic stale upstream state."""
    xds = xds.copy(deep=True)
    xds = xds.assign_coords(
        {
            "right_ascension_input": xds["right_ascension"],
            "declination_input": xds["declination"],
        }
    )
    return xds


def _peak_idx(arr: np.ndarray) -> tuple[int, ...]:
    """Return the max-pixel index for arrays with possible NaNs."""
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("No finite pixels available for peak detection.")
    idx = np.unravel_index(np.nanargmax(np.where(valid, arr, np.nan)), arr.shape)
    return tuple(int(i) for i in idx)


def _galactic_peak_offset_arcsec(
    src: xr.Dataset,
    out: xr.Dataset,
) -> tuple[tuple[int, ...], float, float]:
    """Compute output peak index and Galactic offset from source peak world position."""
    src_idx = _peak_idx(src["SKY"].values)
    src_i_l, src_i_m = src_idx[-2], src_idx[-1]
    src_ra = float(src["right_ascension"].values[src_i_l, src_i_m])
    src_dec = float(src["declination"].values[src_i_l, src_i_m])
    src_gal = SkyCoord(src_ra * u.rad, src_dec * u.rad, frame="fk5").transform_to(
        "galactic"
    )
    src_glon = float(src_gal.spherical.lon.to_value(u.rad))
    src_glat = float(src_gal.spherical.lat.to_value(u.rad))

    out_idx = _peak_idx(out["SKY"].values)
    out_i_l, out_i_m = out_idx[-2], out_idx[-1]
    out_glon = float(out["galactic_longitude"].values[out_i_l, out_i_m])
    out_glat = float(out["galactic_latitude"].values[out_i_l, out_i_m])

    # Small-angle lon offset scaled by cos(lat), plus direct lat offset.
    dlon = (
        np.rad2deg(
            abs(np.arctan2(np.sin(out_glon - src_glon), np.cos(out_glon - src_glon)))
            * np.cos(src_glat)
        )
        * 3600.0
    )
    dlat = np.rad2deg(abs(out_glat - src_glat)) * 3600.0
    return out_idx, float(dlon), float(dlat)


def _major_axis_pa_world(
    ds: xr.Dataset, *, lon_name: str, lat_name: str, frac_threshold: float = 0.2
) -> float:
    """Estimate source major-axis PA from world-coordinate second moments."""
    arr = ds["SKY"].isel(time=0, frequency=0, polarization=0).values
    lon = ds[lon_name].values
    lat = ds[lat_name].values

    valid = np.isfinite(arr)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError(
            "No finite pixels available for source-orientation estimate."
        )

    threshold = float(np.nanmax(arr[valid])) * frac_threshold
    mask = valid & (arr >= threshold)
    if np.count_nonzero(mask) < 5:
        raise RuntimeError("Too few source pixels for source-orientation estimate.")

    weights = arr[mask]
    weights = weights / np.sum(weights)
    lon0 = float(np.average(lon[mask], weights=weights))
    lat0 = float(np.average(lat[mask], weights=weights))

    # Tangent-plane approximation around the source centroid.
    dlon = np.arctan2(np.sin(lon - lon0), np.cos(lon - lon0))
    east = dlon * np.cos(lat0)
    north = lat - lat0

    east_sel = east[mask]
    north_sel = north[mask]
    mean_east = float(np.sum(weights * east_sel))
    mean_north = float(np.sum(weights * north_sel))
    de = east_sel - mean_east
    dn = north_sel - mean_north

    c_ee = float(np.sum(weights * de * de))
    c_nn = float(np.sum(weights * dn * dn))
    c_en = float(np.sum(weights * de * dn))
    cov = np.array([[c_ee, c_en], [c_en, c_nn]])

    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    east_v = float(major[0])
    north_v = float(major[1])

    # PA is measured from north toward east.
    pa = float(np.arctan2(east_v, north_v))
    return _wrap_half_turn(pa)


def _wrap_half_turn(angle: float) -> float:
    """Wrap an angle to [-pi/2, +pi/2] for 180-degree-axis symmetry."""
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return angle


def test_peak_world_position_consistent_for_keep_grid_modes() -> None:
    """Peak world position should be consistent in both keep-grid modes."""
    src = _make_point_source_dataset()
    out_keep_false = reproject_to_frame(
        src, "galactic", keep_grid=False, method="interp", order=1
    )
    out_keep_true = reproject_to_frame(
        src, "galactic", keep_grid=True, method="interp", order=1
    )

    pixel_scale_arcsec = max(
        abs(np.rad2deg(src["l"].values[1] - src["l"].values[0]) * 3600.0),
        abs(np.rad2deg(src["m"].values[1] - src["m"].values[0]) * 3600.0),
    )

    _, dlon_false, dlat_false = _galactic_peak_offset_arcsec(src, out_keep_false)
    _, dlon_true, dlat_true = _galactic_peak_offset_arcsec(src, out_keep_true)

    assert dlon_false <= pixel_scale_arcsec
    assert dlat_false <= pixel_scale_arcsec
    assert dlon_true <= pixel_scale_arcsec
    assert dlat_true <= pixel_scale_arcsec


def test_legacy_world_input_alias_coords_are_not_copied_to_output() -> None:
    """Frame-reprojection output should drop stale world-coordinate alias coords."""
    src = _add_legacy_input_world_aliases(_make_point_source_dataset())
    out = reproject_to_frame(src, "galactic", keep_grid=False, method="interp", order=1)

    assert "right_ascension_input" not in out.coords
    assert "declination_input" not in out.coords


def test_include_original_world_coords_are_pixelwise_transform_consistent() -> None:
    """Original-frame coords should be transformed consistently on output pixels."""
    src = _make_point_source_dataset()
    out = reproject_to_frame(
        src,
        "galactic",
        keep_grid=False,
        method="interp",
        order=1,
        include_original_world_coords=True,
    )

    assert "galactic_longitude" in out.coords
    assert "galactic_latitude" in out.coords
    assert "original_right_ascension" in out.coords
    assert "original_declination" in out.coords

    glon = out["galactic_longitude"].values
    glat = out["galactic_latitude"].values
    ora = out["original_right_ascension"].values
    odec = out["original_declination"].values

    back = SkyCoord(glon.ravel() * u.rad, glat.ravel() * u.rad, frame="galactic")
    back_fk5 = back.transform_to("fk5")
    exp_ra = back_fk5.spherical.lon.to_value(u.rad).reshape(glon.shape)
    exp_dec = back_fk5.spherical.lat.to_value(u.rad).reshape(glat.shape)

    dra = np.arctan2(np.sin(ora - exp_ra), np.cos(ora - exp_ra))
    ddec = odec - exp_dec
    max_dra_arcsec = float(np.nanmax(np.abs(np.rad2deg(dra) * 3600.0)))
    max_ddec_arcsec = float(np.nanmax(np.abs(np.rad2deg(ddec) * 3600.0)))

    assert max_dra_arcsec < 0.05
    assert max_ddec_arcsec < 0.05


def test_keep_grid_false_galactic_axes_parallel_image_edges() -> None:
    """For keep_grid=False, Galactic lon/lat should align with image axes."""
    src = _make_point_source_dataset()
    out = reproject_to_frame(src, "galactic", keep_grid=False, method="interp", order=1)

    glon = out["galactic_longitude"].values
    glat = out["galactic_latitude"].values

    # Along l-axis, lon should dominate and lat should be near-constant.
    dlon_l = np.nanmedian(np.abs(np.diff(glon, axis=0)))
    dlat_l = np.nanmedian(np.abs(np.diff(glat, axis=0)))
    # Along m-axis, lat should dominate and lon should be near-constant.
    dlon_m = np.nanmedian(np.abs(np.diff(glon, axis=1)))
    dlat_m = np.nanmedian(np.abs(np.diff(glat, axis=1)))

    cross_ratio_l = float(dlat_l / max(dlon_l, 1e-30))
    cross_ratio_m = float(dlon_m / max(dlat_m, 1e-30))

    assert cross_ratio_l < 0.01
    assert cross_ratio_m < 0.01


def test_reproject_to_frame_reprojects_all_spatial_vars_in_data_group() -> None:
    """Group mode should reproject every spatial variable resolved from data_group."""
    src = _make_point_source_dataset()
    src["MODEL"] = src["SKY"] * 2.0
    src.attrs["data_groups"] = {"base": ["SKY", "MODEL"]}

    out_group = reproject_to_frame(
        src,
        "galactic",
        data_group="base",
        keep_grid=False,
        method="interp",
        order=1,
    )
    out_model_single = reproject_to_frame(
        src,
        "galactic",
        data_var="MODEL",
        data_group=None,
        keep_grid=False,
        method="interp",
        order=1,
    )

    assert "SKY" in out_group.data_vars
    assert "MODEL" in out_group.data_vars
    assert "galactic_longitude" in out_group.coords
    assert "galactic_latitude" in out_group.coords
    assert out_group["SKY"].shape == out_group["MODEL"].shape
    np.testing.assert_allclose(
        out_group["MODEL"].values,
        out_model_single["MODEL"].values,
        rtol=0.0,
        atol=1e-12,
    )


def test_reproject_to_frame_falls_back_to_data_var_when_data_group_missing() -> None:
    """Missing data_group should fall back to single-variable data_var mode."""
    src = _make_point_source_dataset()
    src.attrs["data_groups"] = {"other": ["SKY"]}

    out = reproject_to_frame(
        src,
        "galactic",
        data_var="SKY",
        data_group="base",
        keep_grid=False,
        method="interp",
        order=1,
    )

    assert "SKY" in out.data_vars
    assert "galactic_longitude" in out.coords
    assert "galactic_latitude" in out.coords


def test_beam_pa_change_matches_source_major_axis_rotation() -> None:
    """Beam PA delta should track source-axis rotation for frame conversion."""
    src = _make_point_source_dataset()

    # Replace point source with an off-center anisotropic Gaussian so
    # orientation is measurable.
    l_coord = src["l"].values
    m = src["m"].values
    ll, mm = np.meshgrid(l_coord, m, indexing="ij")
    l0 = l_coord[90]
    m0 = m[40]
    sigma_l = 3e-5
    sigma_m = 8e-5
    sky = np.exp(-0.5 * (((ll - l0) / sigma_l) ** 2 + ((mm - m0) / sigma_m) ** 2))
    src["SKY"].values[...] = sky[None, None, None, :, :]
    src["SKY"].attrs["units"] = "Jy/beam"

    # Add beam metadata with PA=0 so delta is directly comparable.
    beam = np.zeros((1, 1, 1, 3), dtype=np.float64)
    beam[..., 0] = np.deg2rad(15.0 / 3600.0)
    beam[..., 1] = np.deg2rad(10.0 / 3600.0)
    beam[..., 2] = 0.0
    src["BEAM_FIT_PARAMS_SKY"] = xr.DataArray(
        beam,
        dims=("time", "frequency", "polarization", "beam_params_label"),
        coords={
            "time": src.coords["time"],
            "frequency": src.coords["frequency"],
            "polarization": src.coords["polarization"],
            "beam_params_label": src.coords["beam_params_label"],
        },
    )

    out = reproject_to_frame(src, "galactic", keep_grid=False, method="interp", order=1)

    pa_src = _major_axis_pa_world(
        src, lon_name="right_ascension", lat_name="declination"
    )
    pa_out = _major_axis_pa_world(
        out, lon_name="galactic_longitude", lat_name="galactic_latitude"
    )
    src_delta = _wrap_half_turn(pa_out - pa_src)

    beam_pa_src = float(
        src["BEAM_FIT_PARAMS_SKY"]
        .sel(beam_params_label="pa")
        .isel(time=0, frequency=0, polarization=0)
        .values
    )
    beam_pa_out = float(
        out["BEAM_FIT_PARAMS_SKY"]
        .sel(beam_params_label="pa")
        .isel(time=0, frequency=0, polarization=0)
        .values
    )
    beam_delta = _wrap_half_turn(beam_pa_out - beam_pa_src)

    # Keep tolerance modest to account for interpolation + moment-estimation noise.
    # In this pipeline, beam-PA update has opposite sign to source-axis world rotation.
    assert abs(np.rad2deg(_wrap_half_turn(beam_delta + src_delta))) < 1.0
