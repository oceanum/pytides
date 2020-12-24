"""
Pytides is small Python package for the analysis and prediction of tides. 
Pytides can be used to extrapolate the tidal behaviour at a given location from its previous behaviour. 
The method used is that of harmonic constituents, in particular as presented by P. Schureman in Special Publication 98. 
The fitting of amplitudes and phases is handled by Scipy's leastsq minimisation function. 
Pytides currently supports the constituents used by NOAA, with plans to add more constituent sets. 
It is therefore possible to use the amplitudes and phases published by NOAA directly, without the need to perform the analysis again (although there may be slight discrepancies for some constituents).
"""

from . import astro, constituent, nodal_corrections, tide

__all__ = ["astro", "constituent", "nodal_corrections", "tide"]
