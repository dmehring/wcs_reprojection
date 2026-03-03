"""Regression tests for additional non-observer-dependent frame families."""

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


def test_reproject_to_geocentric_true_ecliptic_writes_ecliptic_world_coords() -> None:
    """Geocentric true ecliptic reprojection should emit ecliptic coordinates."""
    src = _make_point_source_dataset()
    out = reproject_to_frame(
        src, "geocentrictrueecliptic", keep_grid=False, method="bilinear"
    )

    assert "ecliptic_longitude" in out.coords
    assert "ecliptic_latitude" in out.coords
    assert "right_ascension" not in out.coords
    assert "declination" not in out.coords

    elon = out["ecliptic_longitude"].values
    elat = out["ecliptic_latitude"].values
    assert np.isfinite(elon).all()
    assert np.isfinite(elat).all()


def test_include_original_world_coords_for_ecliptic_round_trips_to_fk5() -> None:
    """`original_*` coords should be pixelwise-consistent for ecliptic output."""
    src = _make_point_source_dataset()
    out = reproject_to_frame(
        src,
        "geocentrictrueecliptic",
        keep_grid=False,
        method="bilinear",
        include_original_world_coords=True,
    )

    assert "ecliptic_longitude" in out.coords
    assert "ecliptic_latitude" in out.coords
    assert "original_right_ascension" in out.coords
    assert "original_declination" in out.coords

    elon = out["ecliptic_longitude"].values
    elat = out["ecliptic_latitude"].values
    ora = out["original_right_ascension"].values
    odec = out["original_declination"].values

    back = SkyCoord(
        elon.ravel() * u.rad, elat.ravel() * u.rad, frame="geocentrictrueecliptic"
    )
    back_fk5 = back.transform_to("fk5")
    exp_ra = back_fk5.spherical.lon.to_value(u.rad).reshape(elon.shape)
    exp_dec = back_fk5.spherical.lat.to_value(u.rad).reshape(elat.shape)

    dra = np.arctan2(np.sin(ora - exp_ra), np.cos(ora - exp_ra))
    ddec = odec - exp_dec
    assert float(np.nanmax(np.abs(np.rad2deg(dra) * 3600.0))) < 0.05
    assert float(np.nanmax(np.abs(np.rad2deg(ddec) * 3600.0))) < 0.05
