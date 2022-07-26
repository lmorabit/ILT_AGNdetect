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

##############################################################
## Plotting housekeeping

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['axes.labelsize'] = 'large'
## set up some colours
n = 255
mycols = plt.cm.viridis(np.linspace(0, 1,n))
mycols_m = plt.cm.magma(np.linspace(0, 1,n))
mycols_b = make_my_cmap()
## make a plots directory if it doesn't exist
plots_dir = '/home/xswt42/Dropbox/Documents/papers/agn_id/plots'
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

hightb = 'High '+'$T_b$'
hightb_peak = 'High '+'$T_{b,\mathrm{peak}}$'
hightb_total = 'High '+'$T_{b,\mathrm{total}}$'

## to avoid warnings about log limits on plots
## despite warnings about the limits, everything is plotted accurately
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

##############################################################
## Read in catalogue 

lockman = Table.read( 'Lockman_hr_SED_crossmatched.fits', format='fits' )


##############################################################
## Decide if using peak or combination of peak, total selection

use_peak = True

##############################################################
## Helpful indices

## AGN identifications -- photometric
lotss_optAGN = np.where(lockman['optAGN_LOTSS'] == 1)[0]
lotss_irAGN = np.where(lockman['IRAGN_LOTSS'] == 1)[0]
lotss_xrayAGN = np.where(lockman['XrayAGN_LOTSS'] == 1)[0]
## AGN identifications -- SED fitting
sfg = np.where( lockman['SEDclass'] == 'S')[0]
rqagn = np.where( lockman['SEDclass'] == 'R')[0]
lerg = np.where( lockman['SEDclass'] == 'L')[0]
herg = np.where( lockman['SEDclass'] == 'H')[0]
unclass = np.where( lockman['SEDclass'] == 'U')[0]
radio_excess = np.where( lockman['Excess_flag'] == 1 )[0]  ## classified as radio excess based on Best et al. (2022)
## tb indices
peak_agn = np.where( lockman['Tb_from'] == 1.0 )[0]
total_agn = np.where( lockman['Tb_from'] == 2.0 )[0]
#if use_peak:
#    all_agn = peak_agn
#else:
all_agn = np.unique( np.concatenate([peak_agn,total_agn]) )
unresolved = np.where(lockman['Resolved'] == 'U' )[0]
## useful combinations
sfg_total = np.intersect1d(sfg,total_agn)
sfg_peak = np.intersect1d(sfg,peak_agn)
rqagn_total = np.intersect1d(rqagn,total_agn)
rqagn_peak = np.intersect1d(rqagn,peak_agn)
lerg_total = np.intersect1d(lerg,total_agn)
lerg_peak = np.intersect1d(lerg,peak_agn)
herg_total = np.intersect1d(herg,total_agn)
herg_peak = np.intersect1d(herg,peak_agn)
unclass_total = np.intersect1d(unclass,total_agn)
unclass_peak = np.intersect1d(unclass,peak_agn)
excess_total = np.intersect1d(radio_excess,total_agn)
excess_peak = np.intersect1d(radio_excess,peak_agn)

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
excess_symb='+'
s1 = 8**2
s2 = 8**2

##############################################################################
## Get initial numbers of things first for comparison before any filtering
nopt = len( np.intersect1d( np.where(lockman['optAGN_LOTSS'] == 1)[0], all_agn ) )
nir = len( np.intersect1d( np.where(lockman['IRAGN_LOTSS'] == 1)[0], all_agn ) )
nxray = len( np.intersect1d( np.where(lockman['XrayAGN_LOTSS'] == 1)[0], all_agn ) )
nsfg = len( np.intersect1d( np.where( lockman['SEDclass'] == 'S')[0], all_agn ) )
nrqagn = len( np.intersect1d( np.where( lockman['SEDclass'] == 'R')[0], all_agn ) )
nlerg = len( np.intersect1d( np.where( lockman['SEDclass'] == 'L')[0], all_agn ) )
nherg = len( np.intersect1d( np.where( lockman['SEDclass'] == 'H')[0], all_agn ) )
nunclass = len( np.intersect1d( np.where( lockman['SEDclass'] == 'U')[0], all_agn ) )

nopt_peak = len( np.intersect1d( np.where(lockman['optAGN_LOTSS'] == 1)[0], peak_agn ) )
nir_peak = len( np.intersect1d( np.where(lockman['IRAGN_LOTSS'] == 1)[0], peak_agn ) )
nxray_peak = len( np.intersect1d( np.where(lockman['XrayAGN_LOTSS'] == 1)[0], peak_agn ) )
nsfg_peak = len( np.intersect1d( np.where( lockman['SEDclass'] == 'S')[0], peak_agn ) )
nrqagn_peak = len( np.intersect1d( np.where( lockman['SEDclass'] == 'R')[0], peak_agn ) )
nlerg_peak = len( np.intersect1d( np.where( lockman['SEDclass'] == 'L')[0], peak_agn ) )
nherg_peak = len( np.intersect1d( np.where( lockman['SEDclass'] == 'H')[0], peak_agn ) )
nunclass_peak = len( np.intersect1d( np.where( lockman['SEDclass'] == 'U')[0], peak_agn ) )

nopt_total = len( np.intersect1d( np.where(lockman['optAGN_LOTSS'] == 1)[0], total_agn ) )
nir_total = len( np.intersect1d( np.where(lockman['IRAGN_LOTSS'] == 1)[0], total_agn ) )
nxray_total = len( np.intersect1d( np.where(lockman['XrayAGN_LOTSS'] == 1)[0], total_agn ) )
nsfg_total = len( np.intersect1d( np.where( lockman['SEDclass'] == 'S')[0], total_agn ) )
nrqagn_total = len( np.intersect1d( np.where( lockman['SEDclass'] == 'R')[0], total_agn ) )
nlerg_total = len( np.intersect1d( np.where( lockman['SEDclass'] == 'L')[0], total_agn ) )
nherg_total = len( np.intersect1d( np.where( lockman['SEDclass'] == 'H')[0], total_agn ) )
nunclass_total = len( np.intersect1d( np.where( lockman['SEDclass'] == 'U')[0], total_agn ) )


##############################################################################
## Quality control for sample: compare high resolution and standard resolution flux densities

fig = plt.figure(figsize=(5,5))
plt.scatter( lockman['Total_flux_LOTSS'], lockman['Peak_flux'], marker='.',color='gray', label='All', alpha=0.15 )
plt.errorbar( lockman['Total_flux_LOTSS'][total_agn], lockman['Peak_flux'][total_agn], xerr=lockman['E_Total_flux_LOTSS'][total_agn], yerr=lockman['E_Peak_flux'][total_agn], fmt='none', ecolor=mycols_b[4], elinewidth=0.5, alpha=1, barsabove=False )
plt.scatter( lockman['Total_flux_LOTSS'][total_agn], lockman['Peak_flux'][total_agn], marker='*', facecolor='none', edgecolor=mycols_b[4], s=s2, alpha=0.75, label=hightb_total )
plt.errorbar( lockman['Total_flux_LOTSS'][peak_agn], lockman['Peak_flux'][peak_agn], xerr=lockman['E_Total_flux_LOTSS'][peak_agn], yerr=lockman['E_Peak_flux'][peak_agn], fmt='none', ecolor=mycols_b[1], elinewidth=0.5, alpha=1, barsabove=False )
plt.scatter( lockman['Total_flux_LOTSS'][peak_agn], lockman['Peak_flux'][peak_agn], marker='*', facecolor='none', edgecolor=mycols_b[1], s=s2, alpha=0.75, label=hightb_peak )
plt.plot(np.linspace(0.0001,1),np.linspace(0.0001,1),color='black' )
plt.xscale('log')
plt.yscale('log')
plt.xlabel('LoTSS total flux density [Jy]')
plt.ylabel('ILT peak brightness [Jy beam'+r'$^{-1}$'+']' )
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Flux_comparison.pdf'))

## remove sources with higher peak than total flux ratio, re-do indices -- do this with all AGN so table can be written out
new_idx = np.intersect1d( np.unique(np.concatenate([peak_agn,total_agn])), np.where(lockman['Total_flux_LOTSS'] - lockman['Peak_flux'] > 0)[0] )

print( 'Removing {:d} sources with peak flux greater than total flux'.format( len(np.unique(np.concatenate([peak_agn,total_agn])))-len(new_idx)) )

lotss_optAGN, lotss_irAGN, lotss_xrayAGN, sfg, rqagn, lerg, herg, unclass, radio_excess, peak_agn, total_agn, sfg_peak, sfg_total, rqagn_peak, rqagn_total, lerg_peak, lerg_total, herg_peak, herg_total, unclass_peak, unclass_total, excess_peak, excess_total = redo_indices( new_idx, lotss_optAGN, lotss_irAGN, lotss_xrayAGN, sfg, rqagn, lerg, herg, unclass, radio_excess, peak_agn, total_agn )
all_agn = np.unique( np.concatenate([peak_agn,total_agn]) )

all_seds = np.unique(np.concatenate([sfg,rqagn,lerg,herg,unclass]))
all_seds_peak = np.unique(np.concatenate([sfg_peak,rqagn_peak,lerg_peak,herg_peak,unclass_peak]))
all_seds_total = np.unique(np.concatenate([sfg_total,rqagn_total,lerg_total,herg_total,unclass_total]))

## do this by both peak and total
with open( plots_dir.replace('plots','classification.tex'), 'w') as f:
    f.write( '\\begin{table}\n' )
    f.write( '\\caption{\\label{tab:class} Source classification. The final column shows the number of unique AGN identifications from $T_b$ selection.}\n' )
    f.write( '\\begin{tabular}{lcccc}\n \hline \n \hline \n' )
    f.write( '\\multicolumn{5}{c}{identified with peak and total $T_b$} \\\\ \\hline \n' )
    f.write( 'SED Class & \# & Opt. AGN & IR AGN & Unique $T_b$ AGN \\\\ \\hline\n' )
    n1 = np.intersect1d(sfg_peak,lotss_optAGN)
    n2 = np.intersect1d(sfg_peak,lotss_irAGN)
    sfg_n3 = np.unique(np.concatenate([n1,n2]))
    ## anything that does not have a photometric AGN id is a unique AGN
    f.write( 'SFG & {:d} & {:d} & {:d} & {:d}   \\\\ \n'.format( len(sfg_peak), len(n1), len(n2), len(sfg_peak)-len(sfg_n3) ) )
    n1 = np.intersect1d(unclass_peak,lotss_optAGN)
    n2 = np.intersect1d(unclass_peak,lotss_irAGN)
    unclass_n3 = np.unique(np.concatenate([n1,n2]))
    ## anything that does not have a photometric AGN id is a unique AGN
    f.write( 'Unclass & {:d} & {:d}  & {:d} & {:d}  \\\\ \n'.format( len(unclass_peak), len(n1), len(n2), len(unclass_peak)-len(unclass_n3) ) )
    n1 = np.intersect1d(rqagn_peak,lotss_optAGN)
    n2 = np.intersect1d(rqagn_peak,lotss_irAGN)
    rqagn_n3 = np.unique(np.concatenate([n1,n2,rqagn_peak]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'RQAGN & {:d} & {:d}  & {:d} & {:d} \\\\ \n'.format( len(rqagn_peak), len(n1), len(n2), len(rqagn_peak)-len(rqagn_n3) ) )
    n1 = np.intersect1d(lerg_peak,lotss_optAGN)
    n2 = np.intersect1d(lerg_peak,lotss_irAGN)
    lerg_n3 = np.unique(np.concatenate([n1,n2,lerg_peak]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'LERG & {:d} & {:d}  & {:d} & {:d} \\\\ \n'.format( len(lerg_peak), len(n1), len(n2), len(lerg_peak)-len(lerg_n3) ) )
    n1 = np.intersect1d(herg_peak,lotss_optAGN)
    n2 = np.intersect1d(herg_peak,lotss_irAGN)
    herg_n3 = np.unique(np.concatenate([n1,n2,herg_peak]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'HERG & {:d} & {:d}  & {:d} & {:d}   \\\\ \n'.format( len(herg_peak), len(n1), len(n2), len(herg_peak)-len(herg_n3) ) )
    f.write( '\\textbf{:s}Total{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} \\\\ \\hline \n'.format('{','}','{',len(peak_agn),'}','{', len(np.intersect1d(all_seds_peak,lotss_optAGN)), '}','{', len(np.intersect1d(all_seds_peak,lotss_irAGN)),'}','{',len(sfg_peak)-len(sfg_n3)+len(unclass_peak)-len(unclass_n3),'}'  ) )
    f.write('\\hline \n')
    ## for calculating fractions
    peak_new_identifications = len(sfg_peak)-len(sfg_n3)+len(unclass_peak)-len(unclass_n3)
    f.write( '\\multicolumn{5}{c}{identified with total $T_b$ only} \\\\ \\hline \n' )
    f.write( 'SED Class & \# & Opt. AGN & IR AGN & Unique $T_b$ AGN\\\\ \\hline\n' )
    n1 = np.intersect1d(sfg_total,lotss_optAGN)
    n2 = np.intersect1d(sfg_total,lotss_irAGN)
    sfg_n3 = np.unique(np.concatenate([n1,n2]))
    ## anything that does not have a photometric AGN id is a unique AGN
    f.write( 'SFG & {:d} & {:d} & {:d} & {:d}   \\\\ \n'.format( len(sfg_total), len(n1), len(n2), len(sfg_total)-len(sfg_n3) ) )
    n1 = np.intersect1d(unclass_total,lotss_optAGN)
    n2 = np.intersect1d(unclass_total,lotss_irAGN)
    unclass_n3 = np.unique(np.concatenate([n1,n2]))
    ## anything that does not have a photometric AGN id is a unique AGN
    f.write( 'Unclass & {:d} & {:d}  & {:d} & {:d}  \\\\ \n'.format( len(unclass_total), len(n1), len(n2), len(unclass_total)-len(unclass_n3) ) )
    n1 = np.intersect1d(rqagn_total,lotss_optAGN)
    n2 = np.intersect1d(rqagn_total,lotss_irAGN)
    rqagn_n3 = np.unique(np.concatenate([n1,n2,rqagn_total]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'RQAGN & {:d} & {:d}  & {:d} & {:d} \\\\ \n'.format( len(rqagn_total), len(n1), len(n2), len(rqagn_total)-len(rqagn_n3) ) )
    n1 = np.intersect1d(lerg_total,lotss_optAGN)
    n2 = np.intersect1d(lerg_total,lotss_irAGN)
    lerg_n3 = np.unique(np.concatenate([n1,n2,lerg_total]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'LERG & {:d} & {:d}  & {:d} & {:d} \\\\ \n'.format( len(lerg_total), len(n1), len(n2), len(lerg_total)-len(lerg_n3) ) )
    n1 = np.intersect1d(herg_total,lotss_optAGN)
    n2 = np.intersect1d(herg_total,lotss_irAGN)
    herg_n3 = np.unique(np.concatenate([n1,n2,herg_total]))
    ## anything that does not have either SED or photometric ID is a unique AGN
    f.write( 'HERG & {:d} & {:d}  & {:d} & {:d}   \\\\ \n'.format( len(herg_total), len(n1), len(n2), len(herg_total)-len(herg_n3) ) )
    f.write( '\\textbf{:s}Total{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} & \\textbf{:s}{:d}{:s} \\\\ \\hline \n'.format('{','}','{',len(total_agn),'}','{', len(np.intersect1d(all_seds_total,lotss_optAGN)), '}','{', len(np.intersect1d(all_seds_total,lotss_irAGN)),'}','{',len(sfg_total)-len(sfg_n3)+len(unclass_total)-len(unclass_n3),'}'  ) )
    ## for calculating fractions
    total_new_identifications = len(sfg_total)-len(sfg_n3)+len(unclass_total)-len(unclass_n3)
    f.write( '\\end{tabular} \n' )
    f.write( '\\end{table} \n' )

total_new_identifications = total_new_identifications + peak_new_identifications
peak_previous_identifications = len(peak_agn)-peak_new_identifications
total_previous_identifications = len(total_agn)+len(peak_agn)-total_new_identifications
print( 'Fraction of peak high Tb sources with AGN identifications: {:f} percent'.format( 100*peak_previous_identifications / ( len(peak_agn)) ) )
print( 'Fraction of total high Tb sources with AGN identifications: {:f} percent'.format( 100*total_previous_identifications / ( len(peak_agn)+len(total_agn)) ) )


#######################################################
## Redshift information
zedvals = lockman['z_best']
mybins = np.linspace(0,7,10)
fsizex = 5
fsizey = 11
sbsize = 0.7
sbsizey = 0.5
myylims=(0,0.64)
tpos = (7,1.1)
fig = plt.figure(figsize=(fsizex,fsizey))
p1 = plt.axes([0.15,0.05,sbsize,sbsizey*fsizex/fsizey])

specz = np.where( lockman['Z_BEST_SOURCE_LOTSS'] == 1. )[0]
photz = np.where( lockman['Z_BEST_SOURCE_LOTSS'] == 0. )[0]

## ALL sources identified as RQAGN
counts, bin_edges = np.histogram( zedvals[np.where(lockman['SEDclass'] == 'R')], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p1.hist( bin_edges[:-1], bin_edges, histtype='step', edgecolor=rqagn_c, linewidth=2.0, weights=counts/total_area, label='All RQAGN' )
## Brightness temperature
## first get overall noramlisation
counts, bin_edges = np.histogram( zedvals[rqagn], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
scalefac = np.max(counts/total_area)
sidx = np.intersect1d(rqagn,photz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p1.hist( bin_edges[:-1], bin_edges, color=rqagn_c, alpha=0.25, weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{phot}}$' )
sidx = np.intersect1d(rqagn,specz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p1.hist( bin_edges[:-1], bin_edges, edgecolor=rqagn_c, facecolor='none', fill=True, hatch='//', weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{spec}}$' )
#p1.text( tpos[0], tpos[1], 'RQAGN', size=16, ha='right' )
p1.set_xlabel('Redshift')
p1.set_ylabel('Density')
p1.set_ylim(myylims)
p1.legend()
p2 = plt.axes([0.15,0.05+sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
## ALL sources identified as SFGs
counts, bin_edges = np.histogram( zedvals[np.where(lockman['SEDclass'] == 'S')], bins=mybins )
scalefac = np.max(counts)
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p2.hist( bin_edges[:-1], bin_edges, histtype='step', edgecolor=sfg_c, linewidth=2.0, weights=counts/total_area, label='All SFGs' )
## Brightness temperature
## first get the overall normalisation
counts, bin_edges = np.histogram( zedvals[sfg], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
scalefac = np.max(counts/total_area)
sidx = np.intersect1d(sfg,photz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p2.hist( bin_edges[:-1], bin_edges, color=sfg_c, alpha=0.25, weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{phot}}$' )
sidx = np.intersect1d(sfg,specz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p2.hist( bin_edges[:-1], bin_edges, edgecolor=sfg_c, facecolor='none', fill=True, hatch='//', weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{spec}}$' )
#p2.text( tpos[0], tpos[1], 'SFG', size=16, ha='right' )
p2.set_ylabel('Density')
p2.xaxis.set_visible(False)
p2.set_ylim(myylims)
p2.legend()
p3 = plt.axes([0.15,0.05+2*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
## ALL sources Unclassified
counts, bin_edges = np.histogram( zedvals[np.where(lockman['SEDclass'] == 'U')], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p3.hist( bin_edges[:-1], bin_edges, histtype='step', edgecolor=unclass_c, linewidth=2.0, weights=counts/total_area, label='All Unclassifed' )
## Brightness temperature sources
## first get the overall normalisation
counts, bin_edges = np.histogram( zedvals[unclass], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
scalefac = np.max(counts/total_area)
sidx = np.intersect1d(unclass,photz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p3.hist( bin_edges[:-1], bin_edges, color=unclass_c, alpha=0.25, weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{phot}}$' )
sidx = np.intersect1d(unclass,specz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p3.hist( bin_edges[:-1], bin_edges, edgecolor=unclass_c, facecolor='none', fill=True, hatch='//', weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{spec}}$' )
#p3.text( tpos[0], tpos[1], 'Unclass', size=16, ha='right' )
p3.xaxis.set_visible(False)
p3.set_ylabel('Density')
p3.set_ylim(myylims)
p3.legend()
p4 = plt.axes([0.15,0.05+3*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
## ALL sources
counts, bin_edges = np.histogram( zedvals, bins=mybins )
scalefac = np.max(counts)
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p4.hist( bin_edges[:-1], bin_edges, histtype='step', edgecolor='gray', linewidth=2.0, weights=counts/total_area, label='All Sources' )
## Brighness temperature sources
## first get the overall normalisation
counts, bin_edges = np.histogram( zedvals[all_agn], bins=mybins )
total_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
scalefac = np.max(counts/total_area)
sidx =  np.intersect1d(all_agn,photz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p4.hist( bin_edges[:-1], bin_edges, color='gray', alpha=0.5, weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{phot}}$' )
sidx = np.intersect1d(all_agn,specz)
counts, bin_edges = np.histogram( zedvals[sidx], bins=mybins )
sidx_area = np.sum(counts*(bin_edges[1:]-bin_edges[:-1]))
p4.hist( bin_edges[:-1], bin_edges, edgecolor='gray', facecolor='none', fill=True, hatch='//', weights=counts/sidx_area*scalefac, label='$T_b, z_{\mathrm{spec}}$' )
#p4.text( tpos[0], tpos[1], 'All', size=16, ha='right' )
p4.xaxis.set_visible(False)
p4.set_ylabel('Density')
p4.set_ylim(myylims)
p4.legend()
fig.savefig(os.path.join(plots_dir,'Redshift_info.pdf'))
fig.clear()
plt.close()


##############################################################################
#### DONLEY WEDGE
## Use SERVS and fill in with Swire if SERVS unavailable (only for channel1 and channel2)
chan1 = lockman['ch1_swire_flux_corr_LOTSS']
e_chan1 = lockman['ch1_swire_fluxerr_corr_LOTSS']
chan2 = lockman['ch2_swire_flux_corr_LOTSS']
e_chan2 = lockman['ch2_swire_fluxerr_corr_LOTSS']
chan3 = lockman['ch3_swire_flux_corr_LOTSS']
e_chan3 = lockman['ch3_swire_fluxerr_corr_LOTSS']
chan4 = lockman['ch4_swire_flux_corr_LOTSS']
e_chan4 = lockman['ch4_swire_fluxerr_corr_LOTSS']
chan1[np.where(np.isfinite(lockman['ch1_servs_flux_corr_LOTSS']))] = lockman['ch1_servs_flux_corr_LOTSS'][np.where(np.isfinite(lockman['ch1_servs_flux_corr_LOTSS']))]
e_chan1[np.where(np.isfinite(lockman['ch1_servs_flux_corr_LOTSS']))] = lockman['ch1_servs_fluxerr_corr_LOTSS'][np.where(np.isfinite(lockman['ch1_servs_flux_corr_LOTSS']))]
chan2[np.where(np.isfinite(lockman['ch2_servs_flux_corr_LOTSS']))] = lockman['ch2_servs_flux_corr_LOTSS'][np.where(np.isfinite(lockman['ch2_servs_flux_corr_LOTSS']))]
e_chan2[np.where(np.isfinite(lockman['ch2_servs_flux_corr_LOTSS']))] = lockman['ch2_servs_fluxerr_corr_LOTSS'][np.where(np.isfinite(lockman['ch2_servs_flux_corr_LOTSS']))]

## filter low significance sources
snr_cut = 2.
chan1_snr = np.where( chan1 / e_chan1 >= snr_cut )[0]
chan2_snr = np.where( chan2 / e_chan2 >= snr_cut )[0]
chan3_snr = np.where( chan3 / e_chan3 >= snr_cut )[0]
chan4_snr = np.where( chan4 / e_chan4 >= snr_cut )[0]

ch31_snr = np.intersect1d(chan1_snr,chan3_snr)
ch42_snr = np.intersect1d(chan4_snr,chan2_snr)
ch_snr = np.intersect1d( ch31_snr, ch42_snr )

## handle NaNs and negatives so error messages don't pop up
nan_idx = np.asarray( [i for i in np.arange(0,len(lockman)) if i not in ch_snr] )
chan3[nan_idx] = 1.
chan1[nan_idx] = 1.
chan4[nan_idx] = 1.
chan2[nan_idx] = 1.

## ratios
ch31 = np.log10(chan3/chan1)
ch42 = np.log10(chan4/chan2)
## reset nans
ch31[nan_idx] = np.nan
ch42[nan_idx] = np.nan

## errors
e_ch31 = 0.434 * div_error( chan3, chan1, e_chan3, e_chan1 ) / np.power( 10., ch31 )
e_ch42 = 0.434 * div_error( chan4, chan2, e_chan4, e_chan2 ) / np.power( 10., ch42 )


donley_xcut = [1.2,0.08,0.08,(0.42/1.21),1.2]
donley_ycut = [1.21*1.2+0.27,1.21*0.08+0.27,0.15,0.15,1.21*1.2-0.27]
lacy_xcut = [1.2,-0.1,-0.1,1.2]
lacy_ycut = [-0.2,-0.2,0.34,1.46]

fsizex = 5
fsizey = 11
sbsize = 0.7
sbsizey = 0.5
plxlims = (-1.0,1.2)
plylims = (-1,1.49)
fig = plt.figure(figsize=(fsizex,fsizey))
## bottom figure - hergs / lergs
p1 = plt.axes([0.15,0.05,sbsize,sbsizey*fsizex/fsizey])
p1.scatter( ch31, ch42, marker='.', color='gray', label='All',alpha=0.15 )
## lergs
p1.errorbar( ch31[lerg_total], ch42[lerg_total], xerr=e_ch31[lerg_total], yerr=e_ch42[lerg_total], fmt='none', ecolor=lerg_c, elinewidth=0.5, alpha=1, barsabove=False )
p1.errorbar( ch31[lerg_peak], ch42[lerg_peak], xerr=e_ch31[lerg_peak], yerr=e_ch42[lerg_peak], fmt='none', ecolor=lerg_c, elinewidth=0.5, alpha=1, barsabove=False )
p1.scatter( ch31[lerg_total], ch42[lerg_total], marker=lerg_symb, color='none',edgecolor=lerg_c, s=s2, alpha=0.75 )
p1.scatter( ch31[lerg_peak], ch42[lerg_peak], marker=lerg_symb, color=lerg_c,edgecolor=lerg_c, s=s2, alpha=0.75, label='LERG' )
## hergs
p1.errorbar( ch31[herg_total], ch42[herg_total], xerr=e_ch31[herg_total], yerr=e_ch42[herg_total], fmt='none', ecolor=herg_c, elinewidth=0.5, alpha=1, barsabove=False )
p1.errorbar( ch31[herg_peak], ch42[herg_peak], xerr=e_ch31[herg_peak], yerr=e_ch42[herg_peak], fmt='none', ecolor=herg_c, elinewidth=0.5, alpha=1, barsabove=False )
p1.scatter( ch31[herg_total], ch42[herg_total], marker=herg_symb, color='none',edgecolor=herg_c, s=s2, alpha=0.75 )
p1.scatter( ch31[herg_peak], ch42[herg_peak], marker=herg_symb, color=herg_c,edgecolor=herg_c, s=s2, alpha=0.75, label='HERG' )
p1.plot( donley_xcut, donley_ycut, linewidth=1.5, color=mycols_m[200] )
p1.plot( lacy_xcut, lacy_ycut, color=mycols_m[200], linestyle='dashed' )
p1.set_xlabel('log'+r'$_{10}$'+'$(F_{5.8\mu\mathrm{m}}/F_{3.6\mu\mathrm{m}})$') 
p1.set_ylabel('log'+r'$_{10}$'+'$(F_{8.0\mu\mathrm{m}}/F_{4.5\mu\mathrm{m}})$') 
p1.axes.set_xlim(plxlims)
p1.axes.set_ylim(plylims)
l1 = p1.legend()
dline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linewidth=1.5 )
lline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linestyle='dashed' )
l2 = p1.legend((dline,lline),('Donley','Lacy'),loc='lower right')
p1.add_artist(l1)
## next figure - star forming galaxies
p2 = plt.axes([0.15,0.05+sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
p2.scatter( ch31, ch42, marker='.', color='gray', label='All',alpha=0.15 )
## sfgs
p2.errorbar( ch31[sfg_total], ch42[sfg_total], xerr=e_ch31[sfg_total], yerr=e_ch42[sfg_total], fmt='none', ecolor=sfg_c, elinewidth=0.5, alpha=1, barsabove=False )
p2.errorbar( ch31[sfg_peak], ch42[sfg_peak], xerr=e_ch31[sfg_peak], yerr=e_ch42[sfg_peak], fmt='none', ecolor=sfg_c, elinewidth=0.5, alpha=1, barsabove=False )
p2.scatter( ch31[sfg_total], ch42[sfg_total], marker=sfg_symb, color='none',edgecolor=sfg_c, s=s2*2, alpha=0.75 )
p2.scatter( ch31[sfg_peak], ch42[sfg_peak], marker=sfg_symb, color=sfg_c,edgecolor=sfg_c, s=s2*2, alpha=0.75, label='SFG' )
p2.plot( donley_xcut, donley_ycut, linewidth=1.5, color=mycols_m[200] )
p2.plot( lacy_xcut, lacy_ycut, color=mycols_m[200], linestyle='dashed' )
p2.set_ylabel('log'+r'$_{10}$'+'$(F_{8.0\mu\mathrm{m}}/F_{4.5\mu\mathrm{m}})$') 
p2.axes.set_xlim(plxlims)
p2.axes.set_ylim(plylims)
p2.xaxis.set_visible(False)
l1 = p2.legend()
dline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linewidth=1.5 )
lline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linestyle='dashed' )
l2 = p2.legend((dline,lline),('Donley','Lacy'),loc='lower right')
p2.add_artist(l1)
## next figure - radio quiet AGN
p3 = plt.axes([0.15,0.05+2*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
p3.scatter( ch31, ch42, marker='.', color='gray', label='All',alpha=0.15 )
## rqagns
p3.errorbar( ch31[rqagn_total], ch42[rqagn_total], xerr=e_ch31[rqagn_total], yerr=e_ch42[rqagn_total], fmt='none', ecolor=rqagn_c, elinewidth=0.5, alpha=1, barsabove=False )
p3.errorbar( ch31[rqagn_peak], ch42[rqagn_peak], xerr=e_ch31[rqagn_peak], yerr=e_ch42[rqagn_peak], fmt='none', ecolor=rqagn_c, elinewidth=0.5, alpha=1, barsabove=False )
p3.scatter( ch31[rqagn_total], ch42[rqagn_total], marker=rqagn_symb, color='none',edgecolor=rqagn_c, s=s2*2, alpha=0.75 )
p3.scatter( ch31[rqagn_peak], ch42[rqagn_peak], marker=rqagn_symb, color=rqagn_c,edgecolor=rqagn_c, s=s2*2, alpha=0.75, label='RQAGN' )
p3.plot( donley_xcut, donley_ycut, linewidth=1.5, color=mycols_m[200] )
p3.plot( lacy_xcut, lacy_ycut, color=mycols_m[200], linestyle='dashed' )
p3.set_ylabel('log'+r'$_{10}$'+'$(F_{8.0\mu\mathrm{m}}/F_{4.5\mu\mathrm{m}})$') 
p3.axes.set_xlim(plxlims)
p3.axes.set_ylim(plylims)
p3.xaxis.set_visible(False)
l1 = p3.legend()
dline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linewidth=1.5 )
lline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linestyle='dashed' )
l2 = p3.legend((dline,lline),('Donley','Lacy'),loc='lower right')
p3.add_artist(l1)
## next figure - unclassified
p4 = plt.axes([0.15,0.05+3*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
p4.scatter( ch31, ch42, marker='.', color='gray', label='All',alpha=0.15 )
## unclasss
p4.errorbar( ch31[unclass_total], ch42[unclass_total], xerr=e_ch31[unclass_total], yerr=e_ch42[unclass_total], fmt='none', ecolor=unclass_c, elinewidth=0.5, alpha=1, barsabove=False )
p4.errorbar( ch31[unclass_peak], ch42[unclass_peak], xerr=e_ch31[unclass_peak], yerr=e_ch42[unclass_peak], fmt='none', ecolor=unclass_c, elinewidth=0.5, alpha=1, barsabove=False )
p4.scatter( ch31[unclass_total], ch42[unclass_total], marker=unclass_symb, color='none',edgecolor=unclass_c, s=s2*3, alpha=0.75 )
p4.scatter( ch31[unclass_peak], ch42[unclass_peak], marker=unclass_symb, color=unclass_c,edgecolor=unclass_c, s=s2*3, alpha=0.75, label='Unclass' )
p4.plot( donley_xcut, donley_ycut, linewidth=1.5, color=mycols_m[200] )
p4.plot( lacy_xcut, lacy_ycut, color=mycols_m[200], linestyle='dashed' )
p4.set_ylabel('log'+r'$_{10}$'+'$(F_{8.0\mu\mathrm{m}}/F_{4.5\mu\mathrm{m}})$') 
p4.axes.set_xlim(plxlims)
p4.axes.set_ylim(plylims)
p4.xaxis.set_visible(False)
l1 = p4.legend()
dline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linewidth=1.5 )
lline = matplotlib.lines.Line2D([0],[0], color=mycols_m[200], linestyle='dashed' )
l2 = p4.legend((dline,lline),('Donley','Lacy'),loc='lower right')
p4.add_artist(l1)
#plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Donley.pdf'))
fig.clear()
plt.close()



##############################################################################
## Radio luminosity -- star formation rate

## Radio excess with labels
radio_excess = np.where( lockman['Radio_excess'] >= 0.7 )[0]
def fitline( x, a, b ):
    y = a + b * x
    return(y)
## Definiton in Best et al.
popt = [22.24+0.7, 1.08]

######################
## show peak and total
fig = plt.figure(figsize=(5,5))
plt.plot( lockman['SFR_cons'], np.log10(lockman['LOTSS_power']),'.', color='gray', label='All',alpha=0.5)
plt.plot( lockman['SFR_cons'][total_agn], np.log10(lockman['LOTSS_power'][total_agn]), '*', fillstyle='none',markeredgecolor=mycols_b[4], markersize=9, alpha=0.75, label=hightb_total )
plt.plot( lockman['SFR_cons'][peak_agn], np.log10(lockman['LOTSS_power'][peak_agn]), '*', fillstyle='none',markeredgecolor=mycols_b[1], markersize=9, alpha=0.75, label=hightb_peak )
plt.plot( np.linspace(0,3.5), fitline(np.linspace(0,3.5),popt[0],popt[1]),linestyle='dashed', color=mycols[150], linewidth=2 )
## plot high-Tb with labels ... ?
plt.xlabel('log(SFR '+'$[M_{\odot}$'+' yr'+'$^{-1}]$'+')')
plt.ylabel('log('+'$L_R$'+' '+'[W Hz'+'$^{-1}]$'+')')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Radio_Excess.pdf'))
fig.clear()
plt.close()

#######################
## with classifications
fig = plt.figure(figsize=(5,5))
plt.scatter( lockman['SFR_cons'], np.log10(lockman['LOTSS_power']),marker='.', color='gray', label='All',alpha=0.5)
## LERG
plt.scatter( lockman['SFR_cons'][lerg_total], np.log10(lockman['LOTSS_power'][lerg_total]), marker=lerg_symb, facecolor='none', edgecolor=lerg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][lerg_peak], np.log10(lockman['LOTSS_power'][lerg_peak]), marker=lerg_symb, color=lerg_c, edgecolor=lerg_c, s=s1, alpha=0.75, label='LERG' )
## HERG
plt.scatter( lockman['SFR_cons'][herg_total], np.log10(lockman['LOTSS_power'][herg_total]), marker=herg_symb, facecolor='none', edgecolor=herg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][herg_peak], np.log10(lockman['LOTSS_power'][herg_peak]), marker=herg_symb, color=herg_c, edgecolor=herg_c, s=s1, alpha=0.75, label='HERG' )
## SFG
plt.scatter( lockman['SFR_cons'][sfg_total], np.log10(lockman['LOTSS_power'][sfg_total]), marker=sfg_symb, facecolor='none', edgecolor=sfg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][sfg_peak], np.log10(lockman['LOTSS_power'][sfg_peak]), marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
## RQAGN
plt.scatter( lockman['SFR_cons'][rqagn_total], np.log10(lockman['LOTSS_power'][rqagn_total]), marker=rqagn_symb, facecolor='none', edgecolor=rqagn_c, s=s2, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][rqagn_peak], np.log10(lockman['LOTSS_power'][rqagn_peak]), marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s2, alpha=0.75, label='RQAGN' )
## Unclass
plt.errorbar( np.repeat(5.0,len(unclass_total)), np.log10(lockman['LOTSS_power'][unclass_total]), xerr=np.repeat(0.2,len(unclass_total)), color='none', ecolor=unclass_c, xuplims=True, elinewidth=0.5, capthick=0.5, capsize=1.5 )
plt.errorbar( np.repeat(5.0,len(unclass_peak)), np.log10(lockman['LOTSS_power'][unclass_peak]), xerr=np.repeat(0.2,len(unclass_peak)), color='none', ecolor=unclass_c, xuplims=True, elinewidth=0.5, capthick=0.5, capsize=1.5 )
plt.scatter( np.repeat(5.0,len(unclass_total)), np.log10(lockman['LOTSS_power'][unclass_total]), marker=unclass_symb, facecolor='none', edgecolor=unclass_c, s=s2*2, alpha=0.75 )
plt.scatter( np.repeat(5.0,len(unclass_peak)), np.log10(lockman['LOTSS_power'][unclass_peak]), marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s2*2, alpha=0.75, label='Unclass' )
## draw the ridgeline
ridge_sfr = np.linspace(-3,6)
ridgeline = 22.24 + 1.08*ridge_sfr 
plt.plot(ridge_sfr,ridgeline,linestyle='dashed',color='black')
plt.xlim(-2.7,5.3)
plt.ylim(20.1,29.9)
plt.xlabel('log(SFR '+'$[M_{\odot}$'+' yr'+'$^{-1}]$'+')')
plt.ylabel('log('+'$L_R$'+' '+'[W Hz'+'$^{-1}]$'+')')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Radio_Excess_classifications.pdf'))
fig.clear()
plt.close()

## find where they are in relation to the ridgeline
sfg_ridge = 22.24 + 1.08*lockman['SFR_cons'][np.intersect1d(sfg,all_agn)]
rqagn_ridge = 22.24 + 1.08*lockman['SFR_cons'][np.intersect1d(rqagn,all_agn)]
sfg_above = len(np.where(np.log10(lockman['LOTSS_power'][np.intersect1d(sfg,all_agn)]) >= sfg_ridge )[0] )
rqagn_above = len(np.where(np.log10(lockman['LOTSS_power'][np.intersect1d(rqagn,all_agn)]) >= rqagn_ridge )[0] )
print('{:d}/{:d} ({:.02f}%) SFG are above the ridgeline.'.format(sfg_above,len(np.intersect1d(sfg,all_agn)), 100*sfg_above/len(np.intersect1d(sfg,all_agn))))
print('{:d}/{:d} ({:.02f}%) RQAGN are above the ridgeline.'.format(rqagn_above,len(np.intersect1d(rqagn,all_agn)), 100*rqagn_above/len(np.intersect1d(rqagn,all_agn))))


##############################################################################
## BELOW THIS LINE USE ONLY UNRESOLVED SOURCES 

## remove resolved sources
print( 'Removing {:d} resolved sources identified as AGN with peak flux density'.format( len(peak_agn)-len(np.intersect1d(unresolved,peak_agn)) ))
print( 'Removing {:d} resolved sources identified as AGN with total flux density'.format( len(total_agn)-len(np.intersect1d(unresolved,total_agn)) ))

lotss_optAGN, lotss_irAGN, lotss_xrayAGN, sfg, rqagn, lerg, herg, unclass, radio_excess, peak_agn, total_agn, sfg_peak, sfg_total, rqagn_peak, rqagn_total, lerg_peak, lerg_total, herg_peak, herg_total, unclass_peak, unclass_total, excess_peak, excess_total = redo_indices( unresolved, lotss_optAGN, lotss_irAGN, lotss_xrayAGN, sfg, rqagn, lerg, herg, unclass, radio_excess, peak_agn, total_agn )

all_agn = np.unique(np.concatenate([peak_agn,total_agn]))

##############################################################################
## STAR FORMATION RATES


## assume that the ILT peak flux is AGN in the unclassified sources
## subtract that from LoTSS total and convert the residual to SFR
lotss_sub_peak = lockman['Total_flux_LOTSS'] - lockman['Peak_flux']
e_lotss_sub_peak = add_sub_error( lockman['E_Total_flux_LOTSS'], lockman['E_Peak_flux'])

AGN_lum = radio_power(lockman['Peak_flux'],lockman['z_best'])
e_AGN_lum = radio_power( lockman['E_Peak_flux'], lockman['z_best'])
SFR_lum = radio_power(lotss_sub_peak,lockman['z_best'])
e_SFR_lum = radio_power( e_lotss_sub_peak, lockman['z_best'] )
## Smith et al 2021:
## log10(L150) = (0.9+-0.01)*log10(SFR)+(0.33+-0.04)*log10(M/10^10)+22.22+-0.02
## log10(SFR) = ( log10(L150) - (22.22+-0.02) - (0.33+-0.04)*log10(M/10^10) ) / (0.9+-0.01)

SFR_lum[np.where(SFR_lum < 0)] = np.nan
nan_idx = np.unique(np.concatenate([np.where(np.isnan(SFR_lum))[0], np.where(np.isnan(lockman['Mass_cons']))[0]]))
SFR_lum[nan_idx] = 1.
lockman['Mass_cons'][nan_idx] = 1.
sfr = ( np.log10(SFR_lum) - 22.22 - 0.33*np.log10(lockman['Mass_cons']) ) / 0.9
## re-set the nans
SFR_lum[nan_idx] = np.nan
lockman['Mass_cons'][nan_idx] = np.nan
sfr[nan_idx] = np.nan

## find where they are away from the ridgeline
sfg_ridge = 22.24 + 1.08*lockman['SFR_cons'][np.intersect1d(sfg,all_agn)]
rqagn_ridge = 22.24 + 1.08*lockman['SFR_cons'][np.intersect1d(rqagn,all_agn)]
unclass_ridge = 22.24 + 1.08*sfr[np.intersect1d(unclass,all_agn)]  ## Unclass don't have SFR in the catalogue
sfg_above = len(np.where(np.log10(lockman['LOTSS_power'][np.intersect1d(sfg,all_agn)]) >= sfg_ridge )[0] )
rqagn_above = len(np.where(np.log10(lockman['LOTSS_power'][np.intersect1d(rqagn,all_agn)]) >= rqagn_ridge )[0] )
unclass_above = len(np.where(np.log10(lockman['LOTSS_power'][np.intersect1d(unclass,all_agn)]) >= unclass_ridge )[0] )
print('{:d}/{:d} ({:.02f}%) SFG are above the ridgeline.'.format(sfg_above,len(np.intersect1d(sfg,all_agn)), 100*sfg_above/len(np.intersect1d(sfg,all_agn))))
print('{:d}/{:d} ({:.02f}%) RQAGN are above the ridgeline.'.format(rqagn_above,len(np.intersect1d(rqagn,all_agn)), 100*rqagn_above/len(np.intersect1d(rqagn,all_agn))))
print('{:d}/{:d} ({:.02f}%) Unclass are above the ridgeline.'.format(unclass_above,len(np.intersect1d(unclass,all_agn)), 100*unclass_above/len(np.intersect1d(unclass,all_agn))))

#######################
## with classifications
fig = plt.figure(figsize=(5,5))
plt.scatter( lockman['SFR_cons'], np.log10(lockman['LOTSS_power']),marker='.', color='gray', label='All',alpha=0.5)
## LERG
plt.scatter( lockman['SFR_cons'][lerg_total], np.log10(lockman['LOTSS_power'][lerg_total]), marker=lerg_symb, facecolor='none', edgecolor=lerg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][lerg_peak], np.log10(lockman['LOTSS_power'][lerg_peak]), marker=lerg_symb, color=lerg_c, edgecolor=lerg_c, s=s1, alpha=0.75, label='LERG' )
## HERG
plt.scatter( lockman['SFR_cons'][herg_total], np.log10(lockman['LOTSS_power'][herg_total]), marker=herg_symb, facecolor='none', edgecolor=herg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][herg_peak], np.log10(lockman['LOTSS_power'][herg_peak]), marker=herg_symb, color=herg_c, edgecolor=herg_c, s=s1, alpha=0.75, label='HERG' )
## SFG
plt.scatter( lockman['SFR_cons'][sfg_total], np.log10(lockman['LOTSS_power'][sfg_total]), marker=sfg_symb, facecolor='none', edgecolor=sfg_c, s=s1, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][sfg_peak], np.log10(lockman['LOTSS_power'][sfg_peak]), marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
## RQAGN
plt.scatter( lockman['SFR_cons'][rqagn_total], np.log10(lockman['LOTSS_power'][rqagn_total]), marker=rqagn_symb, facecolor='none', edgecolor=rqagn_c, s=s2, alpha=0.75 )
plt.scatter( lockman['SFR_cons'][rqagn_peak], np.log10(lockman['LOTSS_power'][rqagn_peak]), marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s2, alpha=0.75, label='RQAGN' )
## Unclass
plt.scatter( sfr[unclass_total], np.log10(lockman['LOTSS_power'][unclass_total]), marker=unclass_symb, facecolor='none', edgecolor=unclass_c, s=s2*2, alpha=0.75 )
plt.scatter( sfr[unclass_peak], np.log10(lockman['LOTSS_power'][unclass_peak]), marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s2*2, alpha=0.75, label='Unclass' )
## draw the ridgeline
ridge_sfr = np.linspace(-3,6)
ridgeline = 22.24 + 1.08*ridge_sfr 
plt.plot(ridge_sfr,ridgeline,linestyle='dashed',color='black')
plt.xlim(-2.7,5.3)
plt.ylim(20.1,29.9)
plt.xlabel('log(SFR '+'$[M_{\odot}$'+' yr'+'$^{-1}]$'+')')
plt.ylabel('log('+'$L_R$'+' '+'[W Hz'+'$^{-1}]$'+')')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Radio_Excess_classifications_unclassSFR.pdf'))
fig.clear()
plt.close()



##############################################################################
## AGN LUMINOSITIES

mybins = np.linspace(21,29)

########################
## RATIOS OF AGN TO SFR

## uncertainties
rad_lum = radio_power(lockman['Total_flux_LOTSS'],lockman['z_best'])
e_rad_lum = rad_lum - radio_power(lockman['Total_flux']-lockman['E_Total_flux'],lockman['z_best'])


agn_frac = AGN_lum/rad_lum
e_agn_frac = div_error( AGN_lum, rad_lum, e_AGN_lum, e_rad_lum )
agn_sfr_ratio = AGN_lum/SFR_lum
e_agn_sfr_ratio = div_error( AGN_lum, SFR_lum, e_AGN_lum, e_SFR_lum )


## ratio of AGN to total luminosity
fig = plt.figure(figsize=(8,4))
## bottom plot
p1 = plt.axes([0.1,0.15,0.6,0.4])
p1.errorbar( np.log10(rad_lum[rqagn]), agn_frac[rqagn], xerr=0.434*e_rad_lum[rqagn]/rad_lum[rqagn], yerr=e_agn_frac[rqagn], ecolor=rqagn_c, elinewidth=0.5, fmt='none', barsabove=False )
p1.scatter( np.log10(rad_lum[rqagn]), AGN_lum[rqagn]/rad_lum[rqagn], marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s1, alpha=0.75, label='RQAGN' )
p1.errorbar( np.log10(rad_lum[sfg]), agn_frac[sfg], xerr=0.434*e_rad_lum[sfg]/rad_lum[sfg], yerr=e_agn_frac[sfg], ecolor=sfg_c, elinewidth=0.5, fmt='none', barsabove=False )
p1.scatter( np.log10(rad_lum[sfg]), AGN_lum[sfg]/rad_lum[sfg], marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
p1.errorbar( np.log10(rad_lum[unclass]), agn_frac[unclass], xerr=0.434*e_rad_lum[unclass]/rad_lum[unclass], yerr=e_agn_frac[unclass], ecolor=unclass_c, elinewidth=0.5, fmt='none', barsabove=False )
p1.scatter( np.log10(rad_lum[unclass]), AGN_lum[unclass]/rad_lum[unclass], marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s2**1.3, alpha=0.75, label='Unclass' )
p1.hlines(1,22,28,color='black',linestyle='dashed')
p1.set_ylabel(r'$\frac{L_{\mathrm{AGN}}}{L_{\mathrm{total}}}$')
p1.set_xlabel('log('+'$L_R$'+' '+'[W Hz'+'$^{-1}]$'+')')
p1.axes.set_xlim(22.3,27.5)
## top plot
p2 = plt.axes([0.1,0.575,0.6,0.4])
p2.hlines(1,22,28,color='black',linestyle='dashed')
p2.errorbar( np.log10(rad_lum[rqagn]), agn_sfr_ratio[rqagn], xerr=0.434*e_rad_lum[rqagn]/rad_lum[rqagn], yerr=e_agn_sfr_ratio[rqagn], ecolor=rqagn_c, elinewidth=0.5, fmt='none', barsabove=False )
p2.scatter( np.log10(rad_lum[rqagn]), AGN_lum[rqagn]/SFR_lum[rqagn], marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s1, alpha=0.75, label='RQAGN' )
p2.errorbar( np.log10(rad_lum[sfg]), agn_sfr_ratio[sfg], xerr=0.434*e_rad_lum[sfg]/rad_lum[sfg], yerr=e_agn_sfr_ratio[sfg], ecolor=sfg_c, elinewidth=0.5, fmt='none', barsabove=False )
p2.scatter( np.log10(rad_lum[sfg]), AGN_lum[sfg]/SFR_lum[sfg], marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
p2.errorbar( np.log10(rad_lum[unclass]), agn_sfr_ratio[unclass], xerr=0.434*e_rad_lum[unclass]/rad_lum[unclass], yerr=e_agn_sfr_ratio[unclass], ecolor=unclass_c, elinewidth=0.5, fmt='none', barsabove=False )
p2.scatter( np.log10(rad_lum[unclass]), AGN_lum[unclass]/SFR_lum[unclass], marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s2**1.3, alpha=0.75, label='Unclass' )
p2.set_ylabel(r'$\frac{L_{\mathrm{AGN}}}{L_{\mathrm{SF}}}$')
p2.axes.set_xlim(22.3,27.5)
p2.axes.set_yscale('log')
p2.xaxis.set_visible(False)
p2.axes.set_ylim(1e-2,50)
p2.legend(loc=(1.1,0.2))
p3 = plt.axes([0.7,0.15,0.25,0.4])
mybins = np.linspace(0,1,num=10)
rqagnhist, rqagnbins = np.histogram(agn_frac[rqagn],bins=mybins)
rqagnx = rqagnbins[:-1] + 0.5*(rqagnbins[1]-rqagnbins[0])
p3.step(rqagnhist,rqagnx,where="pre",color=rqagn_c,linewidth=1.5)
sfghist, sfgbins = np.histogram(agn_frac[sfg],bins=mybins)
sfgx = sfgbins[:-1] + 0.5*(sfgbins[1]-sfgbins[0])
p3.step(sfghist,sfgx,where="pre",color=sfg_c,linewidth=1.5)
unclasshist, unclassbins = np.histogram(agn_frac[unclass],bins=mybins)
unclassx = unclassbins[:-1] + 0.5*(unclassbins[1]-unclassbins[0])
p3.step(unclasshist,unclassx,where="pre",color=unclass_c,linewidth=1.5)
## median values
xvals = np.linspace(-5,35)
p3.plot(xvals,np.repeat(np.nanmedian(agn_frac[rqagn]),len(xvals)),color=rqagn_c,linestyle='dotted',linewidth=1.)
p3.plot(xvals,np.repeat(np.nanmedian(agn_frac[sfg]),len(xvals)),color=sfg_c,linestyle='dotted',linewidth=1.)
p3.plot(xvals,np.repeat(np.nanmedian(agn_frac[unclass]),len(xvals)),color=unclass_c,linestyle='dotted',linewidth=1.)
p3.axes.set_xlim(0,46)
p3.yaxis.set_visible(False)
p3.set_xlabel('Number')
fig.savefig(os.path.join(plots_dir,'AGN_fraction_by_luminosity.pdf'))
fig.clear()
plt.close()


## print out a table for the median values
med_idx = np.unique(np.concatenate([rqagn,sfg,unclass]))
with open( plots_dir.replace('plots','fractions.tex'), 'w') as f:
    f.write( '\\begin{table}\n \centering \n' )
    f.write( '\\caption{\\label{tab:fracs} Median values of ratios from Fig.~\\ref{fig:agnfracs}.}\n' )
    f.write( '\\begin{tabular}{lc}\n \\hline' )
    f.write( '\\multicolumn{2}{c}{$L_{\mathrm{AGN}}/L_{\mathrm{SF}}$} \\\\ \hline \n' )
    f.write( 'Class & Median\\\\ \\hline\n' )
    f.write( 'RQAGN & {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.nanmedian(agn_sfr_ratio[rqagn]), stats.median_abs_deviation( agn_sfr_ratio[rqagn], scale='normal', nan_policy='omit' ) ) )
    f.write( 'SFG & {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.nanmedian(agn_sfr_ratio[sfg]), stats.median_abs_deviation( agn_sfr_ratio[sfg], scale='normal', nan_policy='omit' ) ) )
    f.write( 'Unclass &  {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.nanmedian(agn_sfr_ratio[unclass]), stats.median_abs_deviation( agn_sfr_ratio[unclass], scale='normal', nan_policy='omit' ) ) )
    f.write( '\\textbf{:s}Total{:s} & \\textbf{:s} {:.2f}$\\pm${:.2f} {:s} \\\\ \hline \hline  \n'.format( '{', '}', '{', np.nanmedian(agn_sfr_ratio[med_idx]), stats.median_abs_deviation( agn_sfr_ratio[med_idx], scale='normal', nan_policy='omit' ), '}' ) )


    f.write( '\\multicolumn{2}{c}{$L_{\mathrm{AGN}}/L_{\mathrm{total}}$} \\\\ \hline \n' )
    f.write( 'Class &  Median\\\\ \\hline\n' )
    f.write( 'RQAGN & {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.median(agn_frac[rqagn]), stats.median_abs_deviation( agn_frac[rqagn], scale='normal' ) ) )
    f.write( 'SFG & {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.median(agn_frac[sfg]), stats.median_abs_deviation( agn_frac[sfg], scale='normal' ) ) )
    f.write( 'Unclass & {:.2f}$\\pm${:.2f} \\\\ \n'.format( np.median(agn_frac[unclass]), stats.median_abs_deviation( agn_frac[unclass], scale='normal' ) ) )
    f.write( '\\textbf{:s}Total{:s} & \\textbf{:s} {:.2f}$\\pm${:.2f} {:s} \\\\ \hline \hline  \n'.format( '{', '}', '{', np.median(agn_frac[med_idx]), stats.median_abs_deviation( agn_frac[med_idx], scale='normal' ), '}' ) )
    f.write( '\\end{tabular} \n' )
    f.write( '\\end{table} \n' )


## fit the points to see if there are any relations with radio luminosity 

def linear( x, m, b ):
    y = m*x + b
    return( y )

with open( plots_dir.replace('plots','agnfracfits.tex'), 'w') as f:
    f.write( '\\begin{table}\n \centering \n' )
    f.write( '\\caption{\\label{tab:fracfits} Fit parameters for $L_{\\textrm{AGN}}/L_{\mathrm{total}}=m \\times $log$_{10}(L_R) + b$ from Fig.~\\ref{fig:agnfracs}.}\n' )
    f.write( '\\begin{tabular}{lcc}\n \\hline \n' )
    f.write( 'Class & $m$ & $b$ \\\\ \\hline\n' )
    fit_agn = np.concatenate([rqagn,sfg,unclass])
    popt, pcov = curve_fit( linear, np.log10(rad_lum[fit_agn]), agn_frac[fit_agn], p0=[0,0] )
    perr = np.sqrt(np.diag(pcov))
    f.write( 'All & ${:.2f}\\pm${:.2f}  & ${:.2f}\\pm${:.2f} \\\\ \n'.format( popt[0], perr[0], popt[1], perr[1] ) )
    popt, pcov = curve_fit( linear, np.log10(rad_lum[rqagn]), agn_frac[rqagn], p0=[0,0] )
    perr = np.sqrt(np.diag(pcov))
    f.write( 'RQAGN & ${:.2f}\\pm${:.2f}  & ${:.2f}\\pm${:.2f} \\\\ \n'.format( popt[0], perr[0], popt[1], perr[1] ) )
    popt, pcov = curve_fit( linear, np.log10(rad_lum[sfg]), agn_frac[sfg], p0=[0,0] )
    perr = np.sqrt(np.diag(pcov))
    f.write( 'SFG & ${:.2f}\\pm${:.2f} & ${:.2f}\\pm${:.2f} \\\\ \n'.format( popt[0], perr[0], popt[1], perr[1] ) )
    popt, pcov = curve_fit( linear, np.log10(rad_lum[unclass]), agn_frac[unclass], p0=[0,0] )
    perr = np.sqrt(np.diag(pcov))
    f.write( 'Unclass &  ${:.2f}\\pm${:.2f}  & ${:.2f}\\pm${:.2f} \\\\ \\hline \n'.format( popt[0], perr[0], popt[1], perr[1] ) )
    f.write( '\\end{tabular} \n' )
    f.write( '\\end{table} \n' )


###########################################################
## comparison with SED fitting

## the plotting limits generate a runtime error, but still plot properly
pl_limsx=(0.06,2e0)
pl_limsy=(0.006,2e0)
fig = plt.figure(figsize=(5,9))
gs = fig.add_gridspec(3,1,hspace=0)
axs = gs.subplots(sharex=True,sharey=True)
axs[0].scatter( agn_frac[rqagn], lockman['AGNfrac_AF'][rqagn], marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s1, alpha=0.75, label='RQAGN' )
axs[0].scatter( agn_frac[sfg], lockman['AGNfrac_AF'][sfg], marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
axs[0].scatter( agn_frac[unclass], lockman['AGNfrac_AF'][unclass], marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s1, alpha=0.75, label='Unclass' )
axs[0].plot( np.linspace(0.006,2), np.linspace(0.006,2), color='black', linestyle='dashed', alpha=0.5 )
axs[0].set_xlim(pl_limsx)
axs[0].set_ylim(pl_limsy)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel('$f_{\mathrm{AGN}}$')
axs[0].text(0.05,0.9,'AGNfitter',transform=axs[0].transAxes)
#axs[0].legend(loc='lower left')
axs[1].scatter( agn_frac[rqagn], lockman['AGNfrac_CG_S'][rqagn], marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s1, alpha=0.75, label='RQAGN' )
axs[1].scatter( agn_frac[sfg], lockman['AGNfrac_CG_S'][sfg], marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
axs[1].scatter( agn_frac[unclass], lockman['AGNfrac_CG_S'][unclass], marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s1, alpha=0.75, label='Unclass' )
axs[1].plot( np.linspace(0.006,2), np.linspace(0.006,2), color='black', linestyle='dashed', alpha=0.5 )
axs[1].set_xlim(pl_limsx)
axs[1].set_ylim(pl_limsy)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_ylabel('$f_{\mathrm{AGN}}$')
axs[1].text(0.05,0.9,'CIGALE, Skirtor',transform=axs[1].transAxes)
axs[2].scatter( agn_frac[rqagn], lockman['AGNfrac_CG_F'][rqagn], marker=rqagn_symb, color=rqagn_c, edgecolor=rqagn_c, s=s1, alpha=0.75, label='RQAGN' )
axs[2].scatter( agn_frac[sfg], lockman['AGNfrac_CG_F'][sfg], marker=sfg_symb, color=sfg_c, edgecolor=sfg_c, s=s1, alpha=0.75, label='SFG' )
axs[2].scatter( agn_frac[unclass], lockman['AGNfrac_CG_F'][unclass], marker=unclass_symb, color=unclass_c, edgecolor=unclass_c, s=s1, alpha=0.75, label='Unclass' )
axs[2].plot( np.linspace(0.006,2), np.linspace(0.006,2), color='black', linestyle='dashed', alpha=0.5 )
axs[2].set_xlim(pl_limsx)
axs[2].set_ylim(pl_limsy)
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_ylabel('$f_{\mathrm{AGN}}$')
axs[2].set_xlabel('$L_{\mathrm{AGN}}/L_{\mathrm{total}}$')
axs[2].text(0.05,0.9,'CIGALE, Fritz',transform=axs[2].transAxes)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir,'AGN_fractions.pdf'))
fig.clear()
plt.close()



###########################################################
## For the discussion section: AGN luminosities

## final classifications
sed_agn_class = np.where(lockman['RadioAGN_final'] == 1)[0] 
tmp = np.where( np.logical_and( lockman['AGN_final'] == 0, lockman['RadioAGN_final'] == 0 ) )[0]
sed_sfg_class = np.asarray([ i for i in tmp if i not in all_agn ])


mybins = np.linspace(22,28,16)
fsizex = 5
fsizey = 11
sbsize = 0.7
sbsizey = 0.5
myylims=(0,0.79)
tpos = (28,0.65)

###########################################################
## For discussion section
## From Woo & Urry (2002)
woo_urry = Table.read( 'Woo_Urry_2002_Lbols.csv', format='csv' )
RL_idx = np.where( woo_urry['type'] == 'RLQ' )[0]
RQ_idx = [ i for i in np.arange(0,len(woo_urry)) if i not in RL_idx ]

## Convert L_bol to L_rad using the fiducial model in Nims et al. (2015) -- eqn 32
L_winds = 1e-5 * np.power( 10., woo_urry['Lbol'] )*1e-7 / 144e6

fig = plt.figure(figsize=(fsizex,fsizey))
p1 = plt.axes([0.15,0.05,sbsize,sbsizey*fsizex/fsizey])
p1.hist( np.log10(AGN_lum[rqagn]), bins=mybins, color=rqagn_c, alpha=0.75, label='RQAGN', density=True )
p1.hist( np.log10(L_winds[RQ_idx]), bins=mybins, histtype='step', edgecolor='red', linewidth=2.0, density=True, alpha=0.8 )
p1.hist( np.log10(lockman['LOTSS_power'][sed_sfg_class]), bins=mybins, histtype='step', edgecolor='black', linewidth=2.0, density=True, alpha=0.8 )
p1.hist( np.log10(lockman['LOTSS_power'][np.where(lockman['RadioAGN_final'] == 1)]), bins=mybins, histtype='step', edgecolor='blue', linewidth=2.0, density=True, alpha=0.8 )
p1.text( tpos[0], tpos[1], 'RQAGN', size=16, ha='right', color=rqagn_c )
p1.set_xlabel('log('+r'$L_{\mathrm{AGN}}$'+' W Hz'+r'$^{-1}$'+']')
p1.set_ylabel('Density')
p1.set_ylim(myylims)
p2 = plt.axes([0.15,0.05+sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
p2.hist( np.log10(AGN_lum[sfg]), bins=mybins, color=sfg_c, alpha=0.75, label='SFG', density=True )
p2.hist( np.log10(L_winds[RQ_idx]), bins=mybins, histtype='step', edgecolor='red', linewidth=2.0, density=True, alpha=0.8 )
p2.hist( np.log10(lockman['LOTSS_power'][sed_sfg_class]), bins=mybins, histtype='step', edgecolor='black', linewidth=2.0, density=True, alpha=0.8 )
p2.hist( np.log10(lockman['LOTSS_power'][np.where(lockman['RadioAGN_final'] == 1)]), bins=mybins, histtype='step', edgecolor='blue', linewidth=2.0, density=True, alpha=0.8 )
p2.text( tpos[0], tpos[1], 'SFG', size=16, ha='right', color=sfg_c )
p2.set_ylabel('Density')
p2.xaxis.set_visible(False)
p2.set_ylim(myylims)
p3 = plt.axes([0.15,0.05+2*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
p3.hist( np.log10(AGN_lum[unclass]), bins=mybins, color=unclass_c, alpha=0.75, label='Unclass', density=True )
p3.hist( np.log10(L_winds[RQ_idx]), bins=mybins, histtype='step', edgecolor='red', linewidth=2.0, density=True, alpha=0.8 )
p3.hist( np.log10(lockman['LOTSS_power'][sed_sfg_class]), bins=mybins, histtype='step', edgecolor='black', linewidth=2.0, density=True, alpha=0.8 )
p3.hist( np.log10(lockman['LOTSS_power'][np.where(lockman['RadioAGN_final'] == 1)]), bins=mybins, histtype='step', edgecolor='blue', linewidth=2.0, density=True, alpha=0.8 )
p3.text( tpos[0], tpos[1], 'Unclass', size=16, ha='right', color=unclass_c )
p3.xaxis.set_visible(False)
p3.set_ylabel('Density')
p3.set_ylim(myylims)
p4 = plt.axes([0.15,0.05+3*sbsizey*fsizex/fsizey,sbsize,sbsizey*fsizex/fsizey])
## winds -- Nims et al. 2015
p4.hist( np.log10(L_winds[RQ_idx]), bins=mybins, histtype='step', edgecolor='red', linewidth=2.0, density=True, alpha=0.8 )
## star formation
p4.hist( np.log10(lockman['LOTSS_power'][sed_sfg_class]), bins=mybins, histtype='step', edgecolor='black', linewidth=2.0, density=True, alpha=0.8 )
## jets
p4.hist( np.log10(lockman['LOTSS_power'][np.where(lockman['RadioAGN_final'] == 1)]), bins=mybins, histtype='step', edgecolor='blue', linewidth=2.0, density=True, alpha=0.8 )
p4.text( tpos[0]-6, tpos[1], 'Winds', size=12, ha='left', color='red' )
p4.text( tpos[0]-6, tpos[1]+0.06, 'Star formation', size=12, ha='left', color='black' )
p4.text( tpos[0]-6, tpos[1]-0.06, 'Jets', size=12, ha='left', color='blue' )
p4.xaxis.set_visible(False)
p4.set_ylabel('Density')
p4.set_ylim(myylims)
fig.savefig(os.path.join(plots_dir,'AGN_lum_speculation.pdf'))
fig.clear()
plt.close()



## fit the relation
not_nans = np.where( np.isfinite(lockman['Mass_cons']) )[0]

peak_agn = np.intersect1d( not_nans, peak_agn )
total_agn = np.intersect1d( not_nans, total_agn )
all_agn = np.intersect1d( not_nans, all_agn )

## need to also remove hergs and lergs
temp_agn = radio_excess = np.where( lockman['Excess_flag'] == 0 )[0]
all_agn = np.intersect1d( all_agn, temp_agn )

## can run with only spectroscopic redshifts to get fit
#all_agn = np.intersect1d( all_agn, np.where(np.isfinite(lockman['Z_SPEC_LOTSS']))[0] )
#\print( 'Number of spectroscopic: ', len(all_agn) ) 

## subtract because Mass_cons is already logged
xvals = np.log10(AGN_lum)-lockman['Mass_cons']
## there are some negative values, handle them so this doesn't throw an error
zero_idx = np.where( SFR_lum < 0 )[0]
save_vals = SFR_lum[zero_idx]
SFR_lum[zero_idx] = 1.
yvals = np.log10(SFR_lum)-lockman['Mass_cons']
SFR_lum[zero_idx] = save_vals
yvals[zero_idx] = np.nan

popt, pcov = curve_fit( linear, xvals[all_agn], yvals[all_agn], p0=[1,0] )
perr = np.sqrt(np.diag(pcov))
print('Co-evolution fit is: log$_{10}(L_{\\textrm{SFR}}/M_*)= ' + '{:.2f}\pm{:.2f}'.format(popt[0], perr[0]) +'\\times$log$_{10}(L_{\\textrm{AGN}}/M_*) + ' + '{:.2f}\pm{:.2f}$'.format(popt[1], perr[1]))

fig = plt.figure(figsize=(5,5))
## sort and colour by redshift
zvals = lockman['z_best'][all_agn]
nbins = 30
zbins = np.power( 10., np.linspace(np.log10(0.02),np.log10(6.8),nbins) )
zcols = plt.cm.viridis(zbins/np.max(zbins))
col_idx = []
for i in np.arange(0,len(all_agn)):
    idx = np.max( np.where( lockman['z_best'][all_agn[i]] >= zbins )[0] )
    col_idx.append(idx)
#plt.errorbar( np.log10(AGN_lum[all_agn]), np.log10(SFR_lum[all_agn]), xerr=0.434*e_AGN_lum[all_agn]/AGN_lum[all_agn], yerr=0.434*e_SFR_lum[all_agn]/SFR_lum[all_agn], fmt='none', ecolor='gray', elinewidth=0.5, alpha=1, barsabove=False, zorder=0 )
plt.scatter( xvals[all_agn], yvals[all_agn], marker='.', c=zcols[col_idx] )
plt.plot( np.linspace(10,18), linear( np.linspace(10,18), popt[0], popt[1]), color='black', linestyle='dashed' )
plt.plot( np.linspace(10,18), np.linspace(10,18), color='gray', linewidth=0.75 )
cbar = plt.colorbar(ticks=np.linspace(0,1,int(nbins/5)),label='redshift')
cbar.ax.set_yticklabels(np.linspace(0,1,int(nbins/5))*np.max(zbins))
plt.xlim((10.5,16))
plt.ylim((10,16))
plt.xlabel('log$_{10}(L_{\mathrm{AGN}}\,[$'+'W Hz'+'$^{-1}])$' + '-log$_{10}(M_* [M_{\odot}])$')
plt.ylabel('log$_{10}(L_{\mathrm{SF}}\,[$'+'W Hz'+'$^{-1}])$' + '-log$_{10}(M_* [M_{\odot}])$')
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Coevolution.pdf'))
fig.clear()
plt.close()



