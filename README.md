# FlatsimCorr

Tools in Python to compute and evaluate non-tectonic ramp corrections for InSAR time-series processed with the FLATSIM service.
For now, only GIM-based ionospheric corrections are implementaed. Available models: IGS (ESA, JPL, CODE, UPC, IGS weighted mean), JPLD (high-resolution), RGP (around France only).

_flatsim.py_ contains a class for to compute ramp corrections for one FLATSIM dataset.
_ramps.py_ contains tools to compare and evaluate the corrections in the form of ramp time-series.
_utils.py_ contains several tools for analysis and plotting.

**Requirements**
- Python
- Matplotlib
- Numpy
- Scipy
- tqdm
- gdal
- cartopy
- spacepy
- netCDF4

**Install**

No compilation needed, just clone the repository and add it to your $PYTHONPATH variable :

```
    git clone https://github.com/leomarconato/FlatsimCorr.git
    cd FlatsimCorr
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

**Example**

To compute GIM ramps for your FLATSIM data :

```
    from flatsimcorr import flatsim
    flatsim_dir = '/directory/containing/your/flatsim/data'  # should include both TS and DAUX directories
    track = 'D037SUD'
    ts = flatsim(track, datadir=flatsim_dir)                 # create a ts object
    ts.computeTecRampsIGS(model='IGS')                       # compute iono ramps with IGS models ('IGS' weighted mean, 'ESA', 'JPL' or 'UPC')
    ts.computeTecRampsJPLD()                                 # compute iono ramps with high-resolution JPLD model
    ts.computeTecRampsRGP()                                  # compute iono ramps with local RGP model for France
```

A directory with the name `track` will be created in current directory, where computed ramps will be saved (e.g. D037SUD/iono_igs/list_ramp_ra_IGS_ESA.txt).

