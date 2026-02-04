# ome-zarr-rfc3-data

Generate OME-Zarr datasets which are illegal before RFC-3 Unrestricted Dimensions and legal afterwards.

- 1D electrocardiogram time series
  - Previously, <2D datasets were disallowed
  - <https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.electrocardiogram.html#scipy.datasets.electrocardiogram>
- RGB image of an astronaut in XCY order
  - Previously, channel dimensions must precede space dimensions
  - <https://imageio.readthedocs.io/en/stable/user_guide/standardimages.html>
- 6D linear ramp against a 0 background
  - Previously, data could be a maximum of 5D, AND only 1 axis could have unknown/ custom type

## Usage

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `uv run main.py`
- See the data/ directory for outputs
