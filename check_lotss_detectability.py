#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from astropy.table import Table, Column, join
from astropy.cosmology import WMAP9 as cosmo
import astropy
import os
from astropy import units as u
import glob
from astropy.wcs import WCS
from astropy.io import fits
from mocpy import MOC
from astropy.coordinates import SkyCoord
from AGNdetect_functions import *


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

### indices for classifications and plotting setup
sfg_symb="d"
sfg_c=mycols_m[60]
rqagn_symb='s'
rqagn_c=mycols_m[100]
lerg_symb='v'
lerg_c=mycols[50]
herg_symb='^'
herg_c = mycols[90]
unclass_symb='*'
unclass_c = mycols_m[170]

##############################################################
## Detectability 

## Full LoTSS catalogue, available at https://lofar-surveys.org/deepfields_public_lockman.html
lotss_cat = 'lockman_final_cross_match_catalogue-v1.0.fits'
lotss = Table.read( lotss_cat, format='fits' )

## Lockman RMS mosaic
rms_map = '/home/xswt42/data/LOFARVLBI/lofar-surveys/Lockman-rms-mosaic.fits'

detectfile = 'LoTSS_detectability.fits'
if os.path.exists(detectfile):
    lotss = Table.read(detectfile,format='fits')
else:
    ## read the local rms value per source
    local_rms = []
    for i in np.arange(0,len(lotss)):
        local_rms.append( get_local_rms_pixel( lotss['RA'][i], lotss['DEC'][i], rms_map ) )
    local_rms = np.asarray(local_rms)
    in_mosaic = np.where( np.isfinite( local_rms) )[0]
    print( 'There are {:d} sources in the rms mosaic'.format(len(in_mosaic)))
    peak_to_local_rms = lotss['Peak_flux'] / local_rms
    ## write this out so it doesn't have to be run again
    lotss.add_column(peak_to_local_rms, name='Detectability_SNR')
    lotss.write(detectfile, format='fits')

## sources in coverage
in_coverage = np.where( lotss['Detectability_SNR'] > 0. )[0]
tmp = len(np.where(lotss['Total_flux'][in_coverage]/lotss['Isl_rms'][in_coverage] >= 5.)[0])
print( '... There are {:d} sources with total / rms >= 5 in the coverage of the high-resolution image.'.format(tmp))

## find the number of detectable sources
detectable = np.where( lotss['Detectability_SNR'] > 5. )[0]
print( '... {:d} sources are detectable'.format(len(detectable)))

## cross-match to the high resolution catalogue
lockman = Table.read( 'Lockman_hr_SED_crossmatched.fits', format='fits' )
lotss_lockman = join( lotss, lockman, keys='Source_Name', join_type='left' )

## Find fraction of detected vs. detectable sources
detectable = np.where( lotss_lockman['Detectability_SNR'].filled(-99) > 5. )[0]
detected = np.where( np.isfinite(lotss_lockman['Total_flux_LOTSS'].filled(np.nan)) )[0]
tb_detected = np.where( lotss_lockman['Tb_from'].filled(-99) > 0 )[0]
tb_undetected = np.intersect1d( np.where( lotss_lockman['Tb_from'].filled(-99) == 0 )[0], np.where(lotss_lockman['Total_flux_LOTSS'].mask == False )[0] )

fmin = np.min(lotss_lockman['Total_flux_1'][detected])
fmax = np.max(lotss_lockman['Total_flux_1'][detected])
flux_bins = np.power( 10., np.linspace(np.log10(fmin),np.log10(fmax), num=15) )

det_frac, e_det_frac = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][detectable], lotss_lockman['Total_flux_1'][detected] ) 
tb_det_frac, e_tb_det_frac = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][detectable], lotss_lockman['Total_flux_1'][tb_detected] ) 

pl_bins = (flux_bins[:-1] + (flux_bins[1:]-flux_bins[:-1])/2)*1e3
pl_bins1 = (flux_bins[:-1] + (flux_bins[1:]-flux_bins[:-1])/2.5)*1e3

fig = plt.figure(figsize=(5,5))
plt.errorbar( pl_bins, det_frac, yerr=e_det_frac, fmt='none', ecolor='gray', elinewidth=0.75, barsabove=False )
plt.errorbar( pl_bins1, tb_det_frac, yerr=e_tb_det_frac, fmt='none', ecolor='black', elinewidth=0.75, barsabove=False )
plt.scatter( pl_bins, det_frac, marker='+', linewidth=0.75, color='gray', label='All' )
plt.scatter( pl_bins1, tb_det_frac, marker='*', linewidth=0.75, color='blue', label='High '+'$T_b$' )
plt.xscale('log')
plt.xlabel('$S_{i,\mathrm{LoTSS}}\,[$' + 'mJy]' )
plt.ylabel('Fraction of detected / detectable sources' )
plt.legend(loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'detection_fraction.pdf'))
fig.clear()
plt.close()

## Fractions of sub-samples of entire T_b population
sfg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'S')[0], tb_detected )
rqagn = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'R')[0], tb_detected )
lerg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'L')[0], tb_detected )
herg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'H')[0], tb_detected )
unclass = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'U')[0], tb_detected )

rqagn_det, e_rqagn_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_detected], lotss_lockman['Total_flux_1'][rqagn] )
sfg_det, e_sfg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_detected], lotss_lockman['Total_flux_1'][sfg] )
lerg_det, e_lerg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_detected], lotss_lockman['Total_flux_1'][lerg] )
herg_det, e_herg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_detected], lotss_lockman['Total_flux_1'][herg] )
unclass_det, e_unclass_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_detected], lotss_lockman['Total_flux_1'][unclass] )

fig = plt.figure(figsize=(5,5))
plt.errorbar( pl_bins, herg_det, yerr=e_herg_det, fmt='none', ecolor=herg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, herg_det, marker=herg_symb, linewidth=0.75, color=herg_c, label='HERG' )
plt.plot(pl_bins[np.where(np.isfinite(herg_det))], herg_det[np.where(np.isfinite(herg_det))], linestyle='dashed', alpha=0.5, color=herg_c )
plt.errorbar( pl_bins, lerg_det, yerr=e_lerg_det, fmt='none', ecolor=lerg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, lerg_det, marker=lerg_symb, linewidth=0.75, color=lerg_c, label='LERG' )
plt.plot(pl_bins[np.where(np.isfinite(lerg_det))], lerg_det[np.where(np.isfinite(lerg_det))], linestyle='dashed', alpha=0.5, color=lerg_c )
plt.errorbar( pl_bins, sfg_det, yerr=e_sfg_det, fmt='none', ecolor=sfg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, sfg_det, marker=sfg_symb, linewidth=0.75, color=sfg_c, label='SFG' )
plt.plot(pl_bins[np.where(np.isfinite(sfg_det))], sfg_det[np.where(np.isfinite(sfg_det))], linestyle='dashed', alpha=0.5, color=sfg_c )
plt.errorbar( pl_bins, rqagn_det, yerr=e_rqagn_det, fmt='none', ecolor=rqagn_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, rqagn_det, marker=rqagn_symb, linewidth=0.75, color=rqagn_c, label='RQAGN' )
plt.plot(pl_bins[np.where(np.isfinite(rqagn_det))], rqagn_det[np.where(np.isfinite(rqagn_det))], linestyle='dashed', alpha=0.5, color=rqagn_c )
plt.errorbar( pl_bins, unclass_det, yerr=e_unclass_det, fmt='none', ecolor=unclass_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, unclass_det, marker=unclass_symb, linewidth=0.75, color=unclass_c, label='Unclass' )
plt.plot(pl_bins[np.where(np.isfinite(unclass_det))], unclass_det[np.where(np.isfinite(unclass_det))], linestyle='dashed', alpha=0.5, color=unclass_c )
plt.xscale('log')
plt.ylim((-0.05,1.05))
#plt.yscale('log')
plt.xlabel('$S_{i,\mathrm{LoTSS}}\,[$' + 'mJy]' )
plt.ylabel('Fraction of population' )
plt.legend(loc='upper left')
plt.title('All $T_b$-identified AGN')
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'population_fraction.pdf'))
fig.clear()
plt.close()


## Fractions of sub-samples of entire non-T_b population
n_sfg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'S')[0], tb_undetected )
n_rqagn = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'R')[0], tb_undetected )
n_lerg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'L')[0], tb_undetected )
n_herg = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'H')[0], tb_undetected )
n_unclass = np.intersect1d( np.where( lotss_lockman['SEDclass'] == 'U')[0], tb_undetected )

rqagn_det, e_rqagn_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_undetected], lotss_lockman['Total_flux_1'][n_rqagn] )
sfg_det, e_sfg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_undetected], lotss_lockman['Total_flux_1'][n_sfg] )
lerg_det, e_lerg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_undetected], lotss_lockman['Total_flux_1'][n_lerg] )
herg_det, e_herg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_undetected], lotss_lockman['Total_flux_1'][n_herg] )
unclass_det, e_unclass_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][tb_undetected], lotss_lockman['Total_flux_1'][n_unclass] )

fig = plt.figure(figsize=(5,5))
plt.errorbar( pl_bins, herg_det, yerr=e_herg_det, fmt='none', ecolor=herg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, herg_det, marker=herg_symb, linewidth=0.75, color=herg_c, label='HERG' )
plt.plot(pl_bins[np.where(np.isfinite(herg_det))], herg_det[np.where(np.isfinite(herg_det))], linestyle='dashed', alpha=0.5, color=herg_c )
plt.errorbar( pl_bins, lerg_det, yerr=e_lerg_det, fmt='none', ecolor=lerg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, lerg_det, marker=lerg_symb, linewidth=0.75, color=lerg_c, label='LERG' )
plt.plot(pl_bins[np.where(np.isfinite(lerg_det))], lerg_det[np.where(np.isfinite(lerg_det))], linestyle='dashed', alpha=0.5, color=lerg_c )
plt.errorbar( pl_bins, sfg_det, yerr=e_sfg_det, fmt='none', ecolor=sfg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, sfg_det, marker=sfg_symb, linewidth=0.75, color=sfg_c, label='SFG' )
plt.plot(pl_bins[np.where(np.isfinite(sfg_det))], sfg_det[np.where(np.isfinite(sfg_det))], linestyle='dashed', alpha=0.5, color=sfg_c )
plt.errorbar( pl_bins, rqagn_det, yerr=e_rqagn_det, fmt='none', ecolor=rqagn_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, rqagn_det, marker=rqagn_symb, linewidth=0.75, color=rqagn_c, label='RQAGN' )
plt.plot(pl_bins[np.where(np.isfinite(rqagn_det))], rqagn_det[np.where(np.isfinite(rqagn_det))], linestyle='dashed', alpha=0.5, color=rqagn_c )
plt.errorbar( pl_bins, unclass_det, yerr=e_unclass_det, fmt='none', ecolor=unclass_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins, unclass_det, marker=unclass_symb, linewidth=0.75, color=unclass_c, label='Unclass' )
plt.plot(pl_bins[np.where(np.isfinite(unclass_det))], unclass_det[np.where(np.isfinite(unclass_det))], linestyle='dashed', alpha=0.5, color=unclass_c )
plt.xscale('log')
plt.ylim((-0.05,1.05))
#plt.yscale('log')
plt.xlabel('$S_{i,\mathrm{LoTSS}}\,[$' + 'mJy]' )
plt.ylabel('Fraction of population' )
plt.legend(loc='upper left')
plt.title('All other sources in high-resolution image')
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'population_fraction_nonTb.pdf'))
fig.clear()
plt.close()



## Cumulative fractions of T_b identified vs. entire class of sub-population
all_sfg = np.where( lotss_lockman['SEDclass'] == 'S')[0]
all_rqagn = np.where( lotss_lockman['SEDclass'] == 'R')[0]
all_lerg = np.where( lotss_lockman['SEDclass'] == 'L')[0]
all_herg = np.where( lotss_lockman['SEDclass'] == 'H')[0]
all_unclass = np.where( lotss_lockman['SEDclass'] == 'U')[0]


rqagn_det, e_rqagn_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_rqagn], lotss_lockman['Total_flux_1'][rqagn] )
sfg_det, e_sfg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_sfg], lotss_lockman['Total_flux_1'][sfg] )
lerg_det, e_lerg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_lerg], lotss_lockman['Total_flux_1'][lerg] )
herg_det, e_herg_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_herg], lotss_lockman['Total_flux_1'][herg] )
unclass_det, e_unclass_det = get_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_unclass], lotss_lockman['Total_flux_1'][unclass] )

herg_idx = np.where( np.isfinite( herg_det ) )
herg_cum, e_herg_cum = get_cum_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_herg], lotss_lockman['Total_flux_1'][herg], herg_idx )
lerg_idx = np.where( np.isfinite( lerg_det ) )
lerg_cum, e_lerg_cum = get_cum_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_lerg], lotss_lockman['Total_flux_1'][lerg], lerg_idx )
sfg_idx = np.where( np.isfinite( sfg_det ) )
sfg_cum, e_sfg_cum = get_cum_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_sfg], lotss_lockman['Total_flux_1'][sfg], sfg_idx )
rqagn_idx = np.where( np.isfinite( rqagn_det ) )
rqagn_cum, e_rqagn_cum = get_cum_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_rqagn], lotss_lockman['Total_flux_1'][rqagn], rqagn_idx )
unclass_idx = np.where( np.isfinite( unclass_det ) )
unclass_cum, e_unclass_cum = get_cum_det_fractions( flux_bins, lotss_lockman['Total_flux_1'][all_unclass], lotss_lockman['Total_flux_1'][unclass], unclass_idx )

print('Maximum values of cumulative sum:')
print( 'HERG: ', np.max(herg_cum))
print( 'LERG: ', np.max(lerg_cum))
print( 'SFG: ', np.max(sfg_cum))
print( 'RQAGN: ', np.max(rqagn_cum))
print( 'Unclass: ', np.max(unclass_cum))

fig = plt.figure(figsize=(5,5))
plt.errorbar( pl_bins[herg_idx], herg_cum, yerr=e_herg_cum, fmt='none', ecolor=herg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins[herg_idx], herg_cum, marker=herg_symb, linewidth=0.75, color=herg_c, label='HERG' )
plt.plot(pl_bins[herg_idx], herg_cum, linestyle='dashed', alpha=0.5, color=herg_c )
plt.errorbar( pl_bins[lerg_idx], lerg_cum, yerr=e_lerg_cum, fmt='none', ecolor=lerg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins[lerg_idx], lerg_cum, marker=lerg_symb, linewidth=0.75, color=lerg_c, label='LERG' )
plt.plot(pl_bins[lerg_idx], lerg_cum, linestyle='dashed', alpha=0.5, color=lerg_c )
plt.errorbar( pl_bins[sfg_idx], sfg_cum, yerr=e_sfg_cum, fmt='none', ecolor=sfg_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins[sfg_idx], sfg_cum, marker=sfg_symb, linewidth=0.75, color=sfg_c, label='SFG' )
plt.plot(pl_bins[sfg_idx], sfg_cum, linestyle='dashed', alpha=0.5, color=sfg_c )
plt.errorbar( pl_bins[rqagn_idx], rqagn_cum, yerr=e_rqagn_cum, fmt='none', ecolor=rqagn_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins[rqagn_idx], rqagn_cum, marker=rqagn_symb, linewidth=0.75, color=rqagn_c, label='RQAGN' )
plt.plot(pl_bins[rqagn_idx], rqagn_cum, linestyle='dashed', alpha=0.5, color=rqagn_c )
plt.errorbar( pl_bins[unclass_idx], unclass_cum, yerr=e_unclass_cum, fmt='none', ecolor=unclass_c, elinewidth=0.75, barsabove=False, alpha=0.75 )
plt.scatter( pl_bins[unclass_idx], unclass_cum, marker=unclass_symb, linewidth=0.75, color=unclass_c, label='Unclass' )
plt.plot(pl_bins[unclass_idx], unclass_cum, linestyle='dashed', alpha=0.5, color=unclass_c )
plt.xscale('log')
plt.ylim((0.001,2.))
plt.yscale('log')
plt.xlabel('$S_{i,\mathrm{LoTSS}}\,[$' + 'mJy]' )
plt.ylabel('Cumulative sum of fraction of $T_b$-identified AGN' )
#plt.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'fractions_by_population.pdf'))
fig.clear()
plt.close()
