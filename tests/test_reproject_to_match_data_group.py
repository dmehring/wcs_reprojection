"""Regression tests for `reproject_to_match` data-group behavior."""

from __future__ import annotations

import numpy as np
import xarray as xr
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
