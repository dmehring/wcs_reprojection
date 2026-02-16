# WCS Reprojection

Lightweight utilities for WCS-aware reprojection of XRADIO image datasets using
`astropy` + `reproject`, with dask-parallelized plane processing.

## Install (editable)

```bash
python -m pip install -e .
```

## Install (editable, notebook/example dependencies)

```bash
python -m pip install -e ".[notebook]"
```

## Usage (basic)

```python
from xradio.image import open_image
from wcs_reproject import reproject_to_match, reproject_to_frame

src = open_image("/path/to/source.zarr")
tgt = open_image("/path/to/target.zarr")

# Reproject to match target grid/WCS
out = reproject_to_match(src, tgt)

# Reproject to a new sky frame (keep same grid)
out2 = reproject_to_frame(src, "galactic", keep_grid=True)
```
