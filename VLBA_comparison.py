#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from astropy.table import Table, Column, join
from scipy.optimize import curve_fit
from astropy.cosmology import WMAP9 as cosmo
import astropy
## WMAP9 is Hinshaw et al. 2013, H_0=69.3, Omega=0.287
from AGNdetect_functions import *
import os
from astropy import units as u
from scipy import stats
import glob
import astropy.units as u
from mocpy import MOC
from astropy.coordinates import Angle, match_coordinates_sky, SkyCoord


##############################################################
## Plotting housekeeping

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['axes.labelsize'] = 'large'
## set up some colours
n = 255
mycols = plt.cm.viridis(np.linspace(0, 1,n))
mycols_m = plt.cm.magma(np.linspace(0, 1,n))
## make a plots directory if it doesn't exist
plots_dir = '/home/xswt42/Dropbox/Documents/papers/agn_id/plots'
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

hightb = 'High '+'$T_b$'
hightb_peak = 'High '+'$T_{b,peak}$'
hightb_total = 'High '+'$T_{b,total}$'

##############################################################
## Decide if using peak or combination of peak, total selection

use_peak = False

##############################################################
## Read in catalogue 

lockman = Table.read( 'Lockman_hr_SED_crossmatched.fits', format='fits' )

## tb indices
peak_agn = np.where( lockman['Tb_from'] == 1.0 )[0]
total_agn = np.where( lockman['Tb_from'] == 2.0 )[0]
if use_peak:
    all_agn = peak_agn
else:
    all_agn = np.unique( np.concatenate([peak_agn,total_agn]) )

##############################################################################
## cross-match with VLBA observations

vlba = Table.read('VLBA_Middelberg2013.csv', format='csv')
## vlba detected -- although the number of matches remains unaffected
## flux density units are in uJy
vlba = vlba[np.where(vlba['Si']>0)[0]]

## From A. Deller
epochs_ad = SkyCoord( '10h52m56s +57d29m06s' )
epoch_b = SkyCoord( '10h52m08.8s +57d21m33.8s' )
epoch_c = SkyCoord( '10h51m02s +57d13m50.4' )

## Find total area
md = 15
moc_ad = MOC.from_cone( lon=epochs_ad.ra, lat=epochs_ad.dec, radius=Angle(20., u.arcmin), max_depth=md )
moc_b = MOC.from_cone( lon=epoch_b.ra, lat=epoch_b.dec, radius=Angle(20., u.arcmin), max_depth=md )
moc_c = MOC.from_cone( lon=epoch_c.ra, lat=epoch_c.dec, radius=Angle(20., u.arcmin), max_depth=md )
tmp_moc = moc_ad.union( moc_b )
final_moc = tmp_moc.union( moc_c )
moc_area = final_moc.sky_fraction * np.power( ( 180. / np.pi ), 2. ) * 4. * np.pi ## convert from sr to deg^2

lockman_coords = SkyCoord( lockman['RA'], lockman['DEC'], unit='deg' )

## Find the sources in the region
seps_ad = lockman_coords.separation( epochs_ad )
seps_b = lockman_coords.separation( epoch_b )
seps_c = lockman_coords.separation( epoch_c )
idx1 = np.where( 60.*seps_ad.value <= 20. )[0]
idx2 = np.where( 60.*seps_b.value <= 20. )[0]
idx3 = np.where( 60.*seps_c.value <= 20. )[0]

in_region = np.unique( np.concatenate([idx1, idx2, idx3]) )

print( 'There are {:d} LoTSS sources in the region.'.format(len(in_region)))
print( '.... {:d} are Peak high-T_b sources'.format(len(np.intersect1d(peak_agn,in_region))))
print( '.... {:d} are Total high-T_b sources'.format(len(np.intersect1d(total_agn,in_region))))

## cross-match the catalogues
vlba_coords = SkyCoord( vlba['RA'], vlba['Dec'], unit='deg' )
idx, d2d, d3d = match_coordinates_sky( vlba_coords, lockman_coords )
idxv, d2dv, d3dv = match_coordinates_sky( lockman_coords, vlba_coords )
## adjusting this upwards to account for astrometry offsets only yields one more sources at 6 arcsec
sep_constraint = d2d <= 2.0 * u.arcsec
match_idx = idx[sep_constraint]
sep_constraint = d2dv <= 2.0 * u.arcsec
match_idxv = idxv[sep_constraint]

print( 'There are {:d} matches, {:d} of which are Peak high-T_b sources, out of {:d} VLBA detections.'.format(len(match_idx), len(np.intersect1d(match_idx,peak_agn)), len(vlba) ) )
print( 'There are {:d} matches, {:d} of which are Total high-T_b sources, out of {:d} VLBA detections.'.format(len(match_idx), len(np.intersect1d(match_idx,total_agn)), len(vlba) ) )

s610_fluxes = vlba['S610'].filled(0)

## using spectral indices, calculate values at 144 MHz
S144_predicted = []
vlba_vla_ratio = []
for i in np.arange(0,len(vlba)):
    gmrt_flux = float(s610_fluxes[i])
    vla_flux = float(vlba['SVLA'][i])
    specidx = vlba['alpha'][-1]
    S144_predicted.append( gmrt_flux * np.power( 144./610., specidx) )
    vlba_vla_ratio.append( vlba['Si'][i]/vla_flux )
S144_predicted = np.asarray(S144_predicted)
vlba_vla_ratio = np.asarray(vlba_vla_ratio)
ilt_predicted = S144_predicted * vlba_vla_ratio

## s \propto nu^alpha 
fig = plt.figure(figsize=(5,5))
mybins = np.linspace( np.min(vlba['alpha']), np.max(vlba['alpha']), num=15 ) 
not_matched = [ i for i in np.arange(0,len(vlba)) if i not in match_idxv ]
plt.hist( vlba['alpha'][not_matched], bins=mybins, alpha=0.5, label='not matched' )
plt.hist( vlba['alpha'][match_idxv], bins=mybins, alpha=0.5, label='matched' )
plt.xlabel('Spectral index')
plt.ylabel('Number')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'vlba_specidx.pdf'))
fig.clear()
plt.close()

## ILT flux density for matched and not matched, for the brightness temp sources
high_tb_in_region = np.intersect1d(in_region, all_agn)
print( 'There are {:d} high-T_b sources in the VLBA region, for a sky density of {:.2f} sources per square degree.'.format( len( high_tb_in_region ), len(high_tb_in_region)/moc_area ) )

print( 'There are {:d} VLBA-detected sources in the VLBA region, for a sky density of {:.2f} sources per square degree.'.format( len( vlba ), len(vlba)/moc_area ) )

lotss_tb = [ i for i in high_tb_in_region if i not in match_idx ]
lotss_match = np.intersect1d(high_tb_in_region,match_idx)

fig = plt.figure(figsize=(5,5))
mybins = np.power( 10., np.linspace( np.log10(0.00014), np.log10(0.5), num=15 ) ) 
plt.hist( lockman['Peak_flux'][lotss_tb], bins=mybins, alpha=0.5, label='not matched', density=True )
plt.hist( lockman['Peak_flux'][lotss_match], bins=mybins, alpha=0.5, label='matched', density=True )
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Density' )
plt.xlabel('$S_{peak}$' + ' [Jy]' )
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'lotss_fluxes.pdf'))
fig.clear()
plt.close()

