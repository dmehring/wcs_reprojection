"""Contract checks for `xradio.image.make_empty_sky_image` frame semantics."""

from __future__ import annotations

import numpy as np
import pytest
from xradio.image import make_empty_sky_image


@pytest.mark.xfail(
    reason=(
        "Pending upstream xradio fix: direction_reference='galactic' should "
        "materialize Galactic world-coordinate matrices on output."
    ),
    strict=False,
)
def test_make_empty_sky_image_galactic_frame_exposes_galactic_world_coords() -> None:
    """Galactic reference should provide Galactic world-coordinate matrices."""
    cell = np.deg2rad(2.0 / 3600.0)
    xds = make_empty_sky_image(
        phase_center=[0.0, 0.0],
        image_size=[64, 64],
        cell_size=[cell, cell],
        frequency_coords=np.array([1.4e9]),
        pol_coords=["I"],
        time_coords=np.array([59000.0]),
        direction_reference="galactic",
        projection="SIN",
        spectral_reference="lsrk",
        do_sky_coords=True,
    )

    assert "galactic_longitude" in xds.coords
    assert "galactic_latitude" in xds.coords
    assert "right_ascension" not in xds.coords
    assert "declination" not in xds.coords
