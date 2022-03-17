## sph_helspec

This python routine extracts spectra for magnetic helicity and energy only using the knowledge of the magnetic field vector distributed over a sphere.
It does not require the magnetic vector potential as an input. It has appications in stellar and solar observations and their numerical models. 
For further information on how this is done please see: https://doi.org/10.1051/0004-6361/202141101. Do cite this paper in case you use this
routine

### Requirements
- To use these python routines, we need the python wrapper for the SHTns library (written in C) to perform vector spherical harmonic transformations. 
The instructions to compile this C library with its python wrapper can be found here: https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/compil.html
with some additional information on how to install its python extension here: https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/python.html.
- Additional python modules might be required, depending on the file format of the magnetic field data.
  -  If the magnetic field data is available in the FITS format (use the sph_hel_spec_fits.py routine). For this the astropy module is required. 
     If not installed, use "pip install astropy" to do so.
  -  If the magnetic field data comes from a Pencil Code (PC) simulation (use the sph_hel_spec_pc.py routine). This uses the pencil python module that
     is a part of PC.
- If you wish to save the data in an hdf5 file, the h5py module is required. If not installed, use "pip install h5py" to do so.
# Usage
To run this routine simply copy the routine to the relevant directory and execute with the 
- python sph_hel_spec_pc.py (for PC) or sph_hel_spec_fits.py (fits data)
- If you only need magnetic energy then execute the relevant routine with the --onlyenergy=True flag.
  For example: python sph_hel_spec_pc.py --onlyenergy=True
- If you wish to save the spectra in an hdf5 file run the routines with the --savespectra=True flag.
  For example: python sph_hel_spec_fits.py --savespectra=True
- If you wish to save the time averaged spectra in .png format run the routines with the --plotspectra=True flag.
  For example: python sph_hel_spec_fits.py --plotpectra=True

