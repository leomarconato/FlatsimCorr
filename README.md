# FlatsimCorr

Tools in Python to compute and evaluate non-tectonic ramp corrections for InSAR time-series processed with the FLATSIM service.
For now, only GIM-based ionospheric corrections are implementaed. Available models: IGS (ESA, JPL, CODE, UPC, IGS weighted mean), JPLD (high-resolution), RGP (around France only).

_flatsim.py_ contains a class for to compute ramp corrections for one FLATSIM dataset.
_ramps.py_ contains tools to compare and evaluate the corrections in the form of ramp time-series.
_utils.py_ contains several tools for analysis and plotting.

**Requirements:**
- Python
- Matplotlib
- Numpy
- Scipy
- tqdm
- gdal
- cartopy
- spacepy

**Install**
No compilation needed, just clone the repository and add it to your $PYTHONPATH variable :

  git clone [https://github.com/leomarconato/FlatsimCorr.git](https://github.com/leomarconato/FlatsimCorr.git)
  cd flatsimcorr
  export PYTHONPATH="$PYTHONPATH:$(pwd)"

