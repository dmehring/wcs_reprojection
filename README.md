# WCS Reprojection

Lightweight utilities for WCS-aware reprojection of XRADIO image datasets using
`astropy` + `reproject`, with dask-parallelized plane processing.

## Install (editable)

```bash
python -m pip install -e .
```

## Usage (basic)

```python
import xarray as xr
from wcs_reproject import reproject_to_match, reproject_to_frame

src = xr.open_zarr("/path/to/source.zarr")
tgt = xr.open_zarr("/path/to/target.zarr")

# Reproject to match target grid/WCS
out = reproject_to_match(src, tgt)

# Reproject to a new sky frame (keep same grid)
out2 = reproject_to_frame(src, "galactic", keep_grid=True)
```
