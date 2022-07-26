#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from astropy.table import Table, Column
from scipy.optimize import curve_fit
from astropy.cosmology import WMAP9 as cosmo
import astropy
## WMAP9 is Hinshaw et al. 2013, H_0=69.3, Omega=0.287
from AGNdetect_functions import *
import os

##############################################################
## Plotting housekeeping

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['axes.labelsize'] = 'large'
## set up some colours
n = 255
mycols = plt.cm.viridis(np.linspace(0, 1,n))
mycols_b = make_my_cmap()
## make a plots directory 
plots_dir = '/home/xswt42/Dropbox/Documents/papers/agn_id/plots'
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

##############################################################
## Read in Lockman Hole data from Sweijen et al. (2022)
## NB: this is not the publicly available catalogue, which doesn't have deconvolved sizes.
## because of smearing, the deconvolved sizes must be used with caution.
## see paper for discussion.

## SNR cut is based on Peak_flux / Isl_rms
lockman_hr = Table.read('catalogue_filtered_full_SNR5_fluxscaled_withoffset_noduplicates_with_lotss.fits', format='fits' )

## get rid of duplicates: keep the one with the lowest noise
remove_idx = []
for sn in lockman_hr['Source_Name_LOTSS']:
    if len(np.where(lockman_hr['Source_Name_LOTSS'] == sn)[0]) > 1:
        tmp_idx = np.where(lockman_hr['Source_Name_LOTSS'] == sn)[0]
        higher_noise = np.where(lockman_hr['Isl_rms'][tmp_idx] == np.max(lockman_hr['Isl_rms'][tmp_idx]))[0]
        if higher_noise[0] == 2:
            remove_idx.append(tmp_idx[0])
            remove_idx.append(tmp_idx[1])
        else:
            remove_idx.append(tmp_idx[higher_noise])
good_idx = np.asarray( [ i for i in np.arange(0,len(lockman_hr)) if i not in remove_idx ] )
lockman_hr = lockman_hr[good_idx]

## Deep Fields SED catalogue, Best et al. (2022)
lockman_sed = Table.read('AGNclasses_Lockman_DR1_final_extended.fits',format='fits')
## rename column for key matching and cross-match with Sweijen et al. 2022
lockman_hr.rename_column('Source_Name_LOTSS','Source_Name')
lockman = astropy.table.join(lockman_hr,lockman_sed,keys='Source_Name',join_type='left')
## set -99 to nan for ease of filtering later on
lockman['Mass_cons'][np.where(lockman['Mass_cons'] == -99)[0]] = np.nan
lockman['SFR_cons'][np.where(lockman['SFR_cons'] == -99)[0]] = np.nan

## Remove sources with no redshift
good_idx = np.where( np.isfinite( lockman['z_best'] ) )[0]
lockman = lockman[good_idx]

##############################################################
## Fit for unresolved sources using the same method as in
## Shimwell et al. 2022 and Shimwell et al. 2019
## to do this, start off with just the isolated sources, which have S_Code = 'S' in PyBSDF catalogues

unres_idx = np.where(lockman['S_Code'] == 'S' )[0]
xvals = lockman['Peak_flux'] / ( 2.*lockman['E_Peak_flux'] ) + lockman['Total_flux'] / ( 2.*lockman['E_Total_flux'] )
yvals = np.log(lockman['Total_flux']/lockman['Peak_flux']) 

unres_fit_idx = fit_for_unresolved( lockman, unres_idx, resolution=0.3, beam_factor=3., mybins=7, myperc=99.9, plots_dir=plots_dir, doplot=False )
print('Number of unresolved:', len(unres_fit_idx), ' which is', len(unres_fit_idx)/len(lockman)*100,' percent of the sample' )

## add column to table
resolved = np.repeat('R',len(lockman))
resolved[unres_fit_idx] = 'U'
lockman.add_column( resolved, name='Resolved')

##############################################################
## Classifications

## AGN identifications -- photometric
lotss_optAGN = np.where(lockman['optAGN_LOTSS'] == 1)[0]
lotss_irAGN = np.where(lockman['IRAGN_LOTSS'] == 1)[0]
lotss_xrayAGN = np.where(lockman['XrayAGN_LOTSS'] == 1)[0]
## AGN identifications -- SED fitting; these are exclusive so can make one column for IDs
##    AGN_final = 1  ##  Radiative AGN
##    AGN_final = 0  ##  Not radiative AGN
##    AGN_final = -1 ##  No robust classification possible
##    RadioAGN_final = 1  ##  Radio-selected AGN
##    RadioAGN_final = 0  ##  Not a radio AGN
##    RadioAGN_final = -1 ##  No robust classification possible
sfg = np.where( np.logical_and( lockman['AGN_final'] == 0, lockman['RadioAGN_final'] == 0 ) )[0]  ## not radiative AGN, not radio-selected AGN
lerg = np.where( np.logical_and( lockman['AGN_final'] == 0, lockman['RadioAGN_final'] == 1 ) )[0]  ## not radiative AGN, radio-selected AGN --> LERG
rqagn = np.where( np.logical_and( lockman['AGN_final'] == 1, lockman['RadioAGN_final'] == 0 ) )[0]  ## radiative AGN, not radio-selected AGN --> RQAGN
herg = np.where( np.logical_and( lockman['AGN_final'] == 1, lockman['RadioAGN_final'] == 1 ) )[0]  ## radiative AGN, radio-selected AGN --> HERG
## there are no unclass if 'and' so use 'or'
unclass = np.where( np.logical_or( lockman['AGN_final'] == -1, lockman['RadioAGN_final'] == -1 ) )[0]  ## no robust classification possible
sedclass = np.repeat('',len(lockman))
sedclass[sfg] = 'SFG'
sedclass[rqagn] = 'RQAGN'
sedclass[lerg] = 'LERG'
sedclass[herg] = 'HERG'
sedclass[unclass] = 'Unclass'
lockman.add_column( sedclass, name='SEDclass' )

##############################################################
## T_b calculations


################
## Step 1: sizes

maj_lim = 0.4
min_lim = 0.3

## where the deconvolved resolution is smaller than the limiting resolution, use the limiting resolution
## See Radcliffe et al. (2018) Eqn. 6 (also Lobanov 2005)
#im_weight = -0.5
im_weight = 0.5
#### WHAT IS GOING ON HERE, CHECK THIS
lim_factor = np.power( 2., 2-0.5*im_weight ) * np.sqrt( np.log(2.)/np.pi * np.log( lockman['Peak_flux']/lockman['Isl_rms'] / ( lockman['Peak_flux']/lockman['Isl_rms'] -1. ) ) )
maj_lim_theory = maj_lim * lim_factor / 60. / 60. ## convert to degrees
min_lim_theory = min_lim * lim_factor / 60. / 60. ## convert to degrees
maj_idx = np.where( lockman['DC_Maj'] < maj_lim_theory )[0]
min_idx = np.where( lockman['DC_Min'] < min_lim_theory )[0]
lockman['DC_Maj'][maj_idx] = maj_lim_theory[maj_idx]
lockman['DC_Min'][min_idx] = maj_lim_theory[min_idx]
## beam solid angle -- function expects units of arcsec
beam_solid_angle = get_beam_solid_angle( lockman['DC_Maj']*60.*60., lockman['DC_Min']*60.*60. )

################
## Step 2: find flux density per solid angle

## flux per solid angle
compact_flux_per_SA_peak = lockman['Peak_flux']*1e3 / beam_solid_angle
compact_flux_per_SA_total = lockman['Total_flux']*1e3 / beam_solid_angle




##############################################################
## Brightness temperature model parameters
T_e = 1e4
alpha=-0.8
ref_freqs = np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9  ## Hz
## and some frequencies for plotting later
freqs_GHz = np.arange( 1e-3, 1e2, 1e-3 )  ## for plotting
freqs = freqs_GHz*1e9  ## Hz


##############################################################
## High T_b sources, assuming nu_0 = 3 GHz

## Get an index of high T_b sources
peak_agn = find_agn( compact_flux_per_SA_peak, T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)
total_agn = find_agn( compact_flux_per_SA_total, T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)

## Use redshift information to find high T_b sources
peak_agn_withz = find_agn_withz( compact_flux_per_SA_peak, lockman['z_best'], T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)
total_agn_withz = find_agn_withz( compact_flux_per_SA_total, lockman['z_best'], T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)

## define what to use -- with z
peak_agn_final = peak_agn_withz
total_agn_final = total_agn_withz
## and combine these
agn_final = np.unique( np.concatenate((peak_agn_final,total_agn_final)))
print('Number of identified AGN: ', len(agn_final), ' which is ', len(agn_final)/len(lockman)*100, ' percent of sources.')
print(len(peak_agn_final), 'are identified via peak brightness')
print(len(total_agn_final), 'are identified via total brightness')

hightb = 'High '+'$T_b$'
hightb_peak = 'High '+'$T_{b,\mathrm{peak}}$'
hightb_total = 'High '+'$T_{b,\mathrm{total}}$'

## add some information to the table
## column for peak (=1) or total (=2) T_b identification
peak_id = np.zeros(len(lockman))
peak_id[total_agn_final] = 2.
peak_id[peak_agn_final] = 1. ## overwrites total 
tb_from = Column( data=peak_id, name='Tb_from' )
## add single column for flux density per solid angle 
compact_flux_per_SA = np.copy(compact_flux_per_SA_total)
compact_flux_per_SA[peak_agn_final] = compact_flux_per_SA_peak[peak_agn_final]
flux_per_SA = Column( data=compact_flux_per_SA, name='Flux_per_SA' )

## add column for total radio power (and uncertainty) in 6" image
lotss_power = radio_power(lockman['Total_flux_LOTSS'],lockman['z_best'])
e_lotss_power = radio_power(lockman['E_Total_flux_LOTSS'],lockman['z_best'])
radio_power = Column( data=lotss_power, name='LOTSS_power' )
e_radio_power = Column( data=e_lotss_power, name='E_LOTSS_power' )
## identify where things are radio excess
excess_idx = np.where( np.log10(lotss_power) >= 22.24 + 0.7 + 1.08*lockman['SFR_cons'] )[0]
excess_col = np.zeros(len(lockman))
excess_col[excess_idx] = 1.
radio_excess = Column( data=excess_col, name='Excess_flag' )

## add all new columns
lockman.add_column( tb_from )
lockman.add_column( flux_per_SA )
lockman.add_column( radio_power )
lockman.add_column( e_radio_power )
lockman.add_column( radio_excess )

## write out the file
tmp = lockman[agn_final]
tmp.write('HighTb_AGN.fits', format='fits',overwrite=True)

## write out a file for visual identification of sources from cutouts
tmp.add_column( np.repeat('',len(tmp)), name='VisID')
idcat = tmp[['Source_Name','VisID']]
idcat.write('HighTB_identifications.csv', format='csv', overwrite=True)

## also write out the full cross-matched catalogue for later use
lockman.write( 'Lockman_hr_SED_crossmatched.fits', format='fits', overwrite=True )

##############################################################
### Reproduce Condon 1992 Fig 4 
fig = plt.figure(figsize=(4.5,6))
## constant brightness temp lines
brightness_temp = np.array([1e-2,1.,1e2,1e4,1e6])
xht = 7e-4 ## for plotting labels
for bt in brightness_temp:
    yvals = const_tb( bt, freqs )*1e3
    plt.plot( freqs_GHz, yvals, color='gray', ls='--' )
    ## get x-value of freqx_GHz where const_tb( bt, freqs*1e3 ) = xht
    idx = np.where( np.abs( yvals - xht ) == np.min(np.abs(yvals-xht) ) )[0]
    tmp = np.log10(freqs_GHz[idx])
    plt_x = 1.5*np.power(10,tmp)
    exp = str(int(np.log10(bt)))
    plt.text( plt_x, xht, '$10^{'+exp+'}$', rotation=62, color='gray', size=12 )
## brightness temp curves
for i in np.arange(len(ref_freqs)):
    rf = ref_freqs[i]
    tau_norm = 1. 
    opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
    tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) ) 
    ## convert to units of plot 
    mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
    plt.plot( freqs_GHz, mJyperasec2, color=mycols[i*20] )
## plot reference line (nu_0 = 3 GHz) thicker, different style
rf = ref_freqs[6]
tau_norm = 1.
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) )
## convert to units of plot 
mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
plt.plot( freqs_GHz, mJyperasec2, color=mycols[6*20], linewidth=3 )
plt.plot( freqs_GHz, mJyperasec2, color='black', linewidth=3, linestyle='dotted' )
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-3,1e2))
plt.ylim((1e-5,1e3))
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density / solid angle [mJy arcsec'+r'$^{-2}$'+']') 
plt.text(2e-3,1e1,'$T_e=10^{:s}$'.format(str(int(np.log10(T_e)))) )
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'condon_fig4.pdf')) 
fig.clear()
plt.close()

##############################################################
### Reproduce Condon 1992 Fig 4 -- vary redshift
fig = plt.figure(figsize=(4.5,6))
## constant brightness temp lines
brightness_temp = np.array([1e-2,1.,1e2,1e4,1e6]) 
xht = 7e-4
for bt in brightness_temp:
    yvals = const_tb( bt, freqs )*1e3
    plt.plot( freqs_GHz, yvals, color='gray', ls='--' )
    ## get x-value of freqx_GHz where const_tb( bt, freqs*1e3 ) = xht
    idx = np.where( np.abs( yvals - xht ) == np.min(np.abs(yvals-xht) ) )[0]
    tmp = np.log10(freqs_GHz[idx])
    plt_x = 1.5*np.power(10,tmp)
    exp = str(int(np.log10(bt)))
    plt.text( plt_x, xht, '$10^{'+exp+'}$', rotation=62, color='gray', size=12 )
## brightness temp curves
## plot reference line (nu_0 = 3 GHz) thicker, different style
rf = ref_freqs[6]
tau_norm = 1.
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) )
## convert to units of plot 
mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
plt.plot( freqs_GHz, mJyperasec2, color=mycols[6*20], linewidth=3 )
plt.plot( freqs_GHz, mJyperasec2, color='black', linewidth=3, linestyle='dotted' )
zvals = np.linspace(0.5, 5, num=6)
## use 3 GHz as the reference frequency
rf = ref_freqs[6]
for i in np.arange(len(zvals)):
    tau_norm = 1. 
    opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
    tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) ) 
    ## convert to units of plot
    mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3 / (1.+zvals[i])
    plt.plot( freqs_GHz, mJyperasec2, color=mycols_b[i], label='$z={:.2f}$'.format(zvals[i]) )
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-3,1e2))
plt.ylim((1e-5,1e3))
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density / solid angle [mJy arcsec'+r'$^{-2}$'+']')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'condon_fig4_redshift.pdf'))
fig.clear()
plt.close()

##############################################################
### Reproduce Condon 1992 Fig 4 -- vary electron temp
fig = plt.figure(figsize=(4.5,6))
## constant brightness temp lines
brightness_temp = np.array([1e-2,1.,1e2,1e4,1e6])
xht = 7e-4
#xht_vals = np.logspace(7e-4,5e-5,num=5)
for bt in brightness_temp:
    yvals = const_tb( bt, freqs )*1e3
    plt.plot( freqs_GHz, yvals, color='gray', ls='--' )
    ## get x-value of freqx_GHz where const_tb( bt, freqs*1e3 ) = xht
    idx = np.where( np.abs( yvals - xht ) == np.min(np.abs(yvals-xht) ) )[0]
    tmp = np.log10(freqs_GHz[idx])
    plt_x = 1.5*np.power(10,tmp)
    exp = str(int(np.log10(bt)))
    plt.text( plt_x, xht, '$10^{'+exp+'}$', rotation=62, color='gray', size=12 )
## brightness temp curves
elec_temps = [np.power(10,4.5),np.power(10,3.5),1e3]
elec_cols = mycols_b[[5,3,1]]
## use 3 GHz as the reference frequency
rf = ref_freqs[6]
tau_norm = 1. 
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
for i in np.arange(0,len(elec_temps)):
    e_temp = elec_temps[i]
    tb = e_temp * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) ) 
    ## convert to units of plot 
    mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
    plt.plot( freqs_GHz, mJyperasec2, color=elec_cols[i], label='$T_e=10^{'+'{:.1f}'.format(np.log10(e_temp))+'}$' )
## plot reference line (nu_0 = 3 GHz) thicker, different style
rf = ref_freqs[6]
tau_norm = 1.
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) )
## convert to units of plot 
mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
plt.plot( freqs_GHz, mJyperasec2, color=mycols[6*20], linewidth=3 )
plt.plot( freqs_GHz, mJyperasec2, color='black', linewidth=3, linestyle='dotted' )
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-3,1e2))
plt.ylim((1e-5,1e3))
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density / solid angle [mJy arcsec'+r'$^{-2}$'+']')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'condon_fig4_Te.pdf'))
fig.clear()
plt.close()

##############################################################
### Reproduce Condon 1992 Fig 4 -- vary spectral index
fig = plt.figure(figsize=(4.5,6))
## constant brightness temp lines
brightness_temp = np.array([1e-2,1.,1e2,1e4,1e6])
xht = 7e-4
for bt in brightness_temp:
    yvals = const_tb( bt, freqs )*1e3
    plt.plot( freqs_GHz, yvals, color='gray', ls='--' )
    ## get x-value of freqx_GHz where const_tb( bt, freqs*1e3 ) = xht
    idx = np.where( np.abs( yvals - xht ) == np.min(np.abs(yvals-xht) ) )[0]
    ## find the closest 10
    tmp = np.log10(freqs_GHz[idx])
    plt_x = 1.5*np.power(10,tmp)
    exp = str(int(np.log10(bt)))
    plt.text( plt_x, xht, '$10^{'+exp+'}$', rotation=62, color='gray', size=12 )
## brightness temp curves
spec_idxs = [-0.9,-0.7,-0.5]
spec_cols = mycols_b[[5,3,1]]
## use 3 GHz as the reference frequency
rf = ref_freqs[6]
tau_norm = 1. 
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
for i in np.arange(0,len(spec_idxs)):
    tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+spec_idxs[i]) ) 
    ## convert to units of plot 
    mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
    plt.plot( freqs_GHz, mJyperasec2, color=spec_cols[i], label=r'$\alpha=$'+'{:.1f}'.format(spec_idxs[i]) )
## plot reference line (nu_0 = 3 GHz) thicker, different style
rf = ref_freqs[6]
tau_norm = 1.
opt_depth_term = 1. - np.exp( - tau_norm * ( freqs / rf )**-2.1 )
tb = T_e * opt_depth_term * ( 1. + 10. * (freqs/1e9)**(0.1+alpha) )
## convert to units of plot 
mJyperasec2 = tb * (freqs)**2. / 1.38e24 * 1e3
plt.plot( freqs_GHz, mJyperasec2, color=mycols[6*20], linewidth=3 )
plt.plot( freqs_GHz, mJyperasec2, color='black', linewidth=3, linestyle='dotted' )
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-3,1e2))
plt.ylim((1e-5,1e3))
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density / solid angle [mJy arcsec'+r'$^{-2}$'+']')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'condon_fig4_alpha.pdf'))
fig.clear()
plt.close()

## radio power vs redshift
fig = plt.figure(figsize=(5,5))
plt.scatter(lockman['z_best'], np.log10(lockman['LOTSS_power']),marker='.', color='gray', label='All',alpha=0.15)
plt.scatter(lockman['z_best'][total_agn_withz], np.log10(lockman['LOTSS_power'][total_agn_withz]), marker='*', color='none',edgecolor=mycols_b[4], s=9**2, alpha=0.75, label=hightb_total)
plt.scatter(lockman['z_best'][peak_agn_withz], np.log10(lockman['LOTSS_power'][peak_agn_withz]), marker='*', color='none',edgecolor=mycols_b[1], s=9**2, alpha=0.75, label=hightb_peak)
## median uncertainties -- are smaller than the points, don't plot
#plt.errorbar( [2.5,2.5], [23,23], yerr=0.434*np.median(lockman['E_LOTSS_power']/lockman['LOTSS_power']), ecolor='gray', elinewidth=0.5 )
plt.xlabel('Redshift '+r'$(z)$')
plt.ylabel('log('+'$L_R$'+' '+'[W Hz'+'$^{-1}]$'+')')
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Radio_power_vs_redshift.pdf'))
fig.clear()
plt.close()

## peak to total flux per solid angle
fig = plt.figure(figsize=(5,5))
linevals = np.linspace(np.min(np.append(compact_flux_per_SA_total,compact_flux_per_SA_peak)),np.max(np.append(compact_flux_per_SA_total,compact_flux_per_SA_peak))) 
plt.plot(linevals,linevals,color='black')
plt.plot(compact_flux_per_SA_total, compact_flux_per_SA_peak,'.',color='gray',label='All',alpha=0.15)
plt.plot(compact_flux_per_SA_total[total_agn_withz], compact_flux_per_SA_peak[total_agn_withz],'*',fillstyle='none',markeredgecolor=mycols_b[4],alpha=0.75,label=hightb_total,markersize=6)
plt.plot(compact_flux_per_SA_total[peak_agn_withz], compact_flux_per_SA_peak[peak_agn_withz],'*',fillstyle='none',markeredgecolor=mycols_b[1],alpha=0.75,label=hightb_peak,markersize=6)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total flux density / solid angle [mJy arcsec'+r'$^{-2}$'+']')
plt.ylabel('Peak brightness / solid angle [mJy beam'+r'$^{-1}$'+' arcsec'+r'$^{-2}$'+']')
lgnd = plt.legend()
lgnd.legendHandles[0]._legmarker.set_markersize(14)
lgnd.legendHandles[1]._legmarker.set_markersize(10)
lgnd.legendHandles[2]._legmarker.set_markersize(10)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'flux_per_solid_angle_comparison.pdf'))
fig.clear()
plt.close()


## ILT to LoTSS flux ratios
lotss_ilt_ratio = lockman['Total_flux_LOTSS'] / lockman['Total_flux']
ratio_errs = div_error( lockman['Total_flux_LOTSS'], lockman['Total_flux'], lockman['E_Total_flux_LOTSS'], lockman['E_Total_flux'] ) 

fig = plt.figure(figsize=(6,5))
plt.plot(lockman['Total_flux_LOTSS'],lotss_ilt_ratio,'.', color='gray', alpha=0.35, label='All' )
plt.errorbar( lockman['Total_flux_LOTSS'][total_agn_withz],lotss_ilt_ratio[total_agn_withz], xerr=lockman['E_Total_flux_LOTSS'][total_agn_withz], yerr=ratio_errs[total_agn_withz], fmt='none', ecolor=mycols_b[4], elinewidth=0.5, alpha=0.75 )
plt.errorbar( lockman['Total_flux_LOTSS'][peak_agn_withz],lotss_ilt_ratio[peak_agn_withz], xerr=lockman['E_Total_flux_LOTSS'][peak_agn_withz], yerr=ratio_errs[peak_agn_withz], fmt='none', ecolor=mycols_b[1], elinewidth=0.5, alpha=0.75 )
plt.plot( lockman['Total_flux_LOTSS'][total_agn_withz],lotss_ilt_ratio[total_agn_withz],'*',fillstyle='none',markeredgecolor=mycols_b[4],markersize=6, alpha=0.75,label=hightb_total)
plt.plot( lockman['Total_flux_LOTSS'][peak_agn_withz],lotss_ilt_ratio[peak_agn_withz],'*',fillstyle='none',markeredgecolor=mycols_b[1],markersize=6, alpha=0.75,label=hightb_peak)
plt.plot(np.linspace(-1,2,num=10),np.repeat(1,10),'--',color='black')
plt.xlim((9e-5,2))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('LoTSS total flux density [Jy]')
plt.ylabel(r'$S_{\mathrm{LoTSS}}/S_{\mathrm{ILT}}$')
lgnd = plt.legend()
lgnd.legendHandles[0]._legmarker.set_markersize(14)
lgnd.legendHandles[1]._legmarker.set_markersize(10)
lgnd.legendHandles[2]._legmarker.set_markersize(10)
plt.tight_layout()
fig.savefig(os.path.join(plots_dir,'Flux_density_ratios.pdf'))
fig.clear()
plt.close()



################################################
## dicotomy in fig 2
sel_idx = np.where( lockman['Total_flux_LOTSS'] > 5e-2 )[0]
tmp = lockman[sel_idx]
high_idx = np.where( lotss_ilt_ratio[sel_idx] > 7 )[0]
low_idx = np.where( lotss_ilt_ratio[sel_idx] < 7 )[0]

## all LERGs and HERGs
## sizes of high flux ratio sources are < 1 arcsec in most cases
## wider distribution for low flux ratio sources




## size histograms?
fig = plt.figure(figsize=(6,5))




