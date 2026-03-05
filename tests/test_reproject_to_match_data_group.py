"""Regression tests for `reproject_to_match` data-group behavior."""

from __future__ import annotations

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.coordinates import SkyCoord
from xradio.image import make_empty_sky_image

from wcs_reproject import reproject_to_match


def _make_dataset(
    *,
    n_l: int = 96,
    n_m: int = 80,
    cell_arcsec: float = 2.0,
) -> xr.Dataset:
    """Create a simple FK5 image dataset with one SKY data variable."""
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

    arr = np.zeros((n_l, n_m), dtype=np.float64)
    arr[n_l // 2, n_m // 2] = 1.0
    xds["SKY"] = xr.DataArray(
        arr[None, None, None, :, :],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            d: xds.coords[d] for d in ("time", "frequency", "polarization", "l", "m")
        },
    )
    xds["SKY"].attrs["units"] = "Jy/pixel"
    return xds


def test_reproject_to_match_reprojects_all_group_spatial_vars() -> None:
    """Group mode should reproject all resolved spatial variables in the group."""
    src = _make_dataset()
    tgt = _make_dataset(n_l=72, n_m=60, cell_arcsec=3.0)
    src["MODEL"] = src["SKY"] * 2.0
    src.attrs["data_groups"] = {"base": ["SKY", "MODEL"]}

    out_group = reproject_to_match(
        src, tgt, data_group="base", data_var="SKY", method="bilinear"
    )
    out_model_single = reproject_to_match(
        src, tgt, data_group=None, data_var="MODEL", method="bilinear"
    )

    assert "SKY" in out_group.data_vars
    assert "MODEL" in out_group.data_vars
    assert out_group["SKY"].shape[-2:] == tgt["SKY"].shape[-2:]
    np.testing.assert_allclose(
        out_group["MODEL"].values,
        out_model_single["MODEL"].values,
        rtol=0.0,
        atol=1e-12,
    )


def test_reproject_to_match_falls_back_to_data_var_when_group_unresolvable() -> None:
    """Unresolvable group selection should fall back to single-variable mode."""
    src = _make_dataset()
    tgt = _make_dataset(n_l=72, n_m=60, cell_arcsec=3.0)
    src.attrs["data_groups"] = {"base": ["NOT_A_DATA_VAR"]}

    out = reproject_to_match(
        src, tgt, data_group="base", data_var="SKY", method="bilinear"
    )

    assert "SKY" in out.data_vars
    assert out["SKY"].shape[-2:] == tgt["SKY"].shape[-2:]


def test_reproject_to_match_include_original_world_coords_transform_consistent() -> (
    None
):
    """Original-frame coords should be added and match Astropy frame transforms."""
    src = _make_dataset()

    # Build a Galactic template grid so output canonical coords are Galactic.
    cell = np.deg2rad(3.0 / 3600.0)
    tgt = make_empty_sky_image(
        phase_center=[0.0, 0.0],
        image_size=[72, 60],
        cell_size=[cell, cell],
        frequency_coords=np.array([1.4e9]),
        pol_coords=["I"],
        time_coords=np.array([59000.0]),
        direction_reference="galactic",
        projection="SIN",
        spectral_reference="lsrk",
        do_sky_coords=True,
    )

    out = reproject_to_match(
        src,
        tgt,
        data_var="SKY",
        method="bilinear",
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


def test_reproject_to_match_uses_dataset_level_csi_when_sky_lacks_csi() -> None:
    """Dataset-level CSI should drive WCS when SKY attrs omit CSI metadata."""
    src = _make_dataset()
    tgt = _make_dataset(n_l=72, n_m=60, cell_arcsec=3.0)

    # Enforce schema-style metadata placement: Dataset-level only.
    src["SKY"].attrs.pop("coordinate_system_info", None)
    tgt["SKY"].attrs.pop("coordinate_system_info", None)
    assert "coordinate_system_info" in src.attrs
    assert "coordinate_system_info" in tgt.attrs

    out = reproject_to_match(src, tgt, data_var="SKY", method="bilinear")

    assert out["SKY"].shape[-2:] == tgt["SKY"].shape[-2:]
    np.testing.assert_allclose(
        out["right_ascension"].values,
        tgt["right_ascension"].values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        out["declination"].values,
        tgt["declination"].values,
        rtol=0.0,
        atol=0.0,
    )


def test_reproject_to_match_dataset_output_has_no_variable_level_csi() -> None:
    """Dataset outputs should keep CSI at Dataset level, not per data variable."""
    src = _make_dataset()
    tgt = _make_dataset(n_l=72, n_m=60, cell_arcsec=3.0)

    out = reproject_to_match(src, tgt, data_var="SKY", method="bilinear")

    assert "coordinate_system_info" in out.attrs
    assert "coordinate_system_info" not in out["SKY"].attrs


def test_reproject_to_match_group_mode_adds_original_world_coords() -> None:
    """Group mode should also add transform-consistent `original_*` coords."""
    src = _make_dataset()
    src["MODEL"] = src["SKY"] * 2.0
    src.attrs["data_groups"] = {"base": ["SKY", "MODEL"]}

    cell = np.deg2rad(3.0 / 3600.0)
    tgt = make_empty_sky_image(
        phase_center=[0.0, 0.0],
        image_size=[72, 60],
        cell_size=[cell, cell],
        frequency_coords=np.array([1.4e9]),
        pol_coords=["I"],
        time_coords=np.array([59000.0]),
        direction_reference="galactic",
        projection="SIN",
        spectral_reference="lsrk",
        do_sky_coords=True,
    )

    out = reproject_to_match(
        src,
        tgt,
        data_group="base",
        data_var="SKY",
        method="bilinear",
        include_original_world_coords=True,
    )

    assert "SKY" in out.data_vars
    assert "MODEL" in out.data_vars
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
