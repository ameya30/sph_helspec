## sph_helspec

This is a numerical implementation in python that extracts spectra for magnetic enery and magnetic helicity only using the
knowledge of the magnetic field vector distributed over a sphere. It has appications in stellar and solar observations and their numerical
models. The formalism for this numerical implementation can be found in this paper: https://doi.org/10.1051/0004-6361/202141101

#Requirements
To use this we need the SHTns library for scalar and vector spherical harmonic transforms. The instructions to compile
this C library with its python wrapper can be found here: https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/compil.html
