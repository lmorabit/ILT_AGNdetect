# ILT_AGNdetect

Notes before starting:
----------------------
- Hardcoded plots_dir variable will need to be changed
- Initial catalogue with deconvolved sizes available on request
- Mosiac of rms noise available on request
- SED fitting results will be publicly available on lofar-surveys.org 


Software requirements:
----------------------

This code was written and run using Python 3.8.10 using the following module versions:
- astropy (4.2)
- matplotlib (3.3.4)
- mocpy (0.10.1)
- scipy (1.6.0)
- numpy (1.20.2)
We cannot guarantee reproducibility if different versions of software are used.

Description of scripts
----------------------

__*initial_catalogue_processing.py*__
    
Script that does initial reading of data, cross-matches to multi-wavelength catalouges, and calculates flux density per solid angle for initial AGN identification. 
    
outputs:

* HighTb_AGN.fits
* HighTB_identifications.csv
* Lockman_hr_SED_crossmatched.fits
* Figures in paper: 1, 2, 3, 4


__*analysis.py*__
Takes the output of the initial_catalogue_processing.py and performs analysis of the cross-matched catalogue. 

outputs:
* Figures in paper: 5, 6, 7, 10, 11, 12, 13

__*check_lotss_detectability.py*__

Finds which sources are detectable in the high-resolution image based on peak LoTSS flux and the rms map, investigates some fractions. 

outputs:
* catalogue with detectability (if this exists, the rms map is not needed)
* Figures in paper: 8, 9, 14

__*VLBA_comparison.py*__

To conduct a comparison with Middelberg et al. (2013).

outputs:
* Figures in paper: A1

__*AGNdetect_functions.py*__

File with functions for import to python scripts


Order of operations
-------------------

1) Run initial_catalogue_processing.py

2) Run analysis.py

3) Run check_lotss_detectability.py

4) Run VLBA_comparison.py


