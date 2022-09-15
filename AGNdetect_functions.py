#!/usr/bin/python3

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy.table import Table 
from scipy.optimize import curve_fit
from astropy.cosmology import WMAP9 as cosmo
import os
from astropy.io import fits
from astropy.wcs import WCS

#% JR comments have a # and %
######################################################################
## Generic helpful functions

def radio_power( obs_flux_Jy, redshift, spectral_index=-0.8):
    flux_cgs = obs_flux_Jy * 1e-23
    DL = cosmo.luminosity_distance(redshift).value * 3.086e24 ## convert to cm
    power_cgs = 4. * np.pi * np.power(DL,2.) * flux_cgs / np.power( 1+redshift, 1.+spectral_index )
    ## convert to W/Hz
    power = power_cgs * 1e-7
    return( power )

def make_my_cmap( ):
    a = np.array((254,235,226))
    b = np.array((252,197,192))
    c = np.array((250,159,181))
    d = np.array((247,104,161))
    e = np.array((221,52,151))
    f = np.array((174,1,126))
    g = np.array((122,1,119))
    a = a / 255.
    b = b / 255.
    c = c / 255.
    d = d / 255.
    e = e / 255.
    f = f / 255.
    g = g / 255.
    a = np.insert(a,len(a),1)
    b = np.insert(b,len(b),1)
    c = np.insert(c,len(c),1)
    d = np.insert(d,len(d),1)
    e = np.insert(e,len(e),1)
    f = np.insert(f,len(f),1)
    g = np.insert(g,len(g),1)
    cmap = np.array((g,f,e,d,c,b,a))
    return(cmap)

######################################################################
## Functions for calculating brightness temp curves

def get_beam_solid_angle( theta1, theta2 ):
    ## for a 2D Gaussian: https://www.cv.nrao.edu/~sransom/web/Ch3.html eqn 3.118
    result =  np.pi / (4. * np.log(2.)) * theta1 * theta2 
    return( result )

def const_tb( T_b, freqs ):
    ## Eqn 3 in this work
    ## for plotting on Condon Fig 4
    result = T_b * freqs**2. / 1.38e24
    return( result )

def get_tb_at_curve_point( reffreqHz, freqs, mycurve ):
    ## find the brightness temperature on a curve from Condon (1992)
    idx = np.where( np.abs( freqs - reffreqHz ) == np.min( np.abs( freqs - reffreqHz ) ) )[0]
    ## flux density converted from mJy to Jy
    T_b = mycurve[idx] * 1e-3 * 1.38e24 / reffreqHz**2.
    return( T_b[0] )

def get_fluxlim_for_beam( T_blim, circ_hpbw, reffreqHz ):
    ## Get the flux limit based on the max brightness temperature, assuming a circular beam
    ## Eqn 3 in this work
    fluxlim = T_blim * reffreqHz**2. / 1.38e24 * get_beam_solid_angle( circ_hpbw, circ_hpbw )
    return( fluxlim )

def lofar_resolution( reffreqHz, longest_bl_km=1989. ):
    ## theoretical resolution of LOFAR based on the longest baseline
    speedoflight = 2.99792458e5 ## km / s
    theta = ( speedoflight / reffreqHz ) / longest_bl_km
    theta_arcsec = theta * 206265
    return( theta_arcsec )

######################################################################
## Functions for processing LOFAR data from catalogue

def make_bins_and_percentiles(xvals,yvals,nbins=10,percent=99.9):
    ## sort the xvals first
    sort_idx = np.argsort( xvals )
    xvals = xvals[sort_idx]
    yvals = yvals[sort_idx]
    ## divide into bins with equal number
    nvals = len(yvals) // nbins
    bin_mids = np.zeros(nbins)
    bin_percs = np.zeros(nbins)
    for i in np.arange(nbins):
        i1 = i*nvals
        i2 = i1 + nvals
        bin_mids[i] = np.median(xvals[i1:i2])
        bin_percs[i] = np.percentile(yvals[i1:i2],percent)
    if i2 < len(yvals):
        bin_mids = np.append(bin_mids,np.median(xvals[i2:]))
        bin_percs = np.append(bin_percs,np.percentile(yvals[i2:],percent))
    return( bin_mids, bin_percs )

def make_even_log_bins_and_percentiles(xvals,yvals,nbins=10,percent=95,minx=0,maxx=0):
    exps = np.log10(xvals)
    if minx > 0:
        binmin = np.log10(minx)
    else:
        binmin = np.min(exps)
    if maxx > 0:
        binmax = np.log10(maxx)
    else:
        binmax = np.max(exps)
    bins = np.power( 10., np.linspace(binmin, binmax, num=nbins) )
    bin_percs = np.zeros(len(bins)-1)
    for i in np.arange(len(bins)-1):
        binvals = yvals[np.where(np.logical_and( xvals >= bins[i], xvals < bins[i+1] ) )]
        if len(binvals) > 0:
            bin_percs[i] = np.percentile(binvals,percent)
    bin_mids = np.power( 10., 0.5*(np.log10(bins[1:]) - np.log10(bins[0:-1]))+np.log10(bins[0:-1]) )
    return( bin_mids, bin_percs )

def sigmoid( x, a, b, c ):
    # Shimwell et al. 2022 Eqn 2
    y = a + b / ( 1. + np.power( x/96.57, c ) )
    return( y )

def find_unresolved( peak, e_peak, total, e_total, coeffs ):
    xvals = peak / (2*e_peak) + total/(2*e_total)
    yvals = np.log( total / peak )
    ysig = sigmoid( xvals, coeffs[0], coeffs[1], coeffs[2] )
    idx = np.where( yvals <= ysig )[0]
    return(idx)

def fit_for_unresolved( mycat, selection_idx, resolution=0.3, beam_factor=3., mybins=12, myperc=99.9, plots_dir='/home/xswt42/Dropbox/Documents/papers/agn_id/plots/', doplot=False):
    ## Following DR2, try to pick out unresolved sources -- combine SNR cut with major axis cut
    xvals = mycat['Peak_flux'] / ( 2.*mycat['E_Peak_flux'] ) + mycat['Total_flux'] / ( 2.*mycat['E_Total_flux'] )    
    yvals = np.log(mycat['Total_flux']/mycat['Peak_flux'])     
    ## Following DR2, try to pick out unresolved sources -- combine SNR cut with major axis cut
    # major axis cut (3 x beam FWHM)
    beam_cut = np.where(mycat['Maj']*60.*60. < resolution*beam_factor )[0]
    ## This is the selection method for DR2
    dr2_idx = np.intersect1d(beam_cut, selection_idx)

    ## 1. make bins
    ## 1a. and find where XX points lie under the given percentile
    bins_even, percs_even = make_even_log_bins_and_percentiles(xvals[dr2_idx],yvals[dr2_idx],nbins=mybins, percent=myperc, maxx=200 )
    ## filter bins with zeros
    nonzero_idx = np.where( percs_even > 0 )[0]
    bins_even = bins_even[nonzero_idx]
    percs_even = percs_even[nonzero_idx]
    ## 3. fit a sigmoid function to these points
    p0 = [0, 1, 2] #% Any motivation for these initial parameters? Or just random?
    popt, pcov = curve_fit( sigmoid, bins_even, percs_even, p0 )
    print( 'fit:', popt )

    if doplot:
        ## plot
        n = 255
        mycols = plt.cm.viridis(np.linspace(0, 1,n))
        mycols_m = plt.cm.magma(np.linspace(0, 1,n))
        fig = plt.figure(figsize=(6,5))
        plt.plot(xvals,yvals,'.',color='gray',label='SNR cut',alpha=0.5)
        plt.plot(xvals[dr2_idx],yvals[dr2_idx],'.',label='Sample for fitting',color=mycols[20],alpha=0.5)
        plt.plot(bins_even,percs_even,'x',label='Bin {:s} percentile'.format(str(myperc)),markersize=12)
        plt_bins = np.power(10.,np.linspace(0,4))
        plt.plot(plt_bins, sigmoid(plt_bins, popt[0], popt[1], popt[2] ), label='Fitted sigmoid'.format(str(myperc)) )
        plt.xscale('log')
        plt.xlabel(r'$\frac{S_p}{2\sigma_p}+\frac{S_i}{2\sigma_i}$')
        plt.ylabel('ln'+r'$(\frac{S_i}{S_p})$')
        plt.xlim((3e0,4e3))
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir,'Unresolved.pdf'))
        fig.clear()
        plt.close()

    ## get the new unresolved index
    unres_fit_idx = find_unresolved( mycat['Peak_flux'], mycat['E_Peak_flux'], mycat['Total_flux'], mycat['E_Total_flux'], popt )
    return( unres_fit_idx )

def find_agn( flux_per_SA, T_e=1e4, rf_array=np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9, freq=144e6, alpha=-0.8 ):
    ## predict brightness temps at frequency
    flux_per_SA_rfs = []
    for rf in rf_array:        
        tau_norm = 1. 
        opt_depth_term = 1. - np.exp( - tau_norm * ( freq / rf )**-2.1 )
        tb = T_e * opt_depth_term * ( 1. + 10. * (freq/1e9)**(0.1+alpha) ) 
        mJyperasec2 = tb * (freq)**2. / 1.38e24 * 1e3
        flux_per_SA_rfs.append(mJyperasec2)
    maxval = np.max(np.array(flux_per_SA_rfs))
    idx = np.where(flux_per_SA >= maxval)[0]
    return( idx )

def find_agn_withz( flux_per_SA, redshift, T_e=1e4, rf_array=np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9, freq=144e6, alpha=-0.8 ):
    ## predict brightness temps at frequency
    flux_per_SA_rfs = []
    for rf in rf_array:        
        tau_norm = 1. 
        opt_depth_term = 1. - np.exp( - tau_norm * ( freq / rf )**-2.1 ) 
        tb = T_e * opt_depth_term * ( 1. + 10. * (freq/1e9)**(0.1+alpha) ) 
        mJyperasec2 = tb * (freq/(1.+redshift))**2. / 1.38e24 * 1e3
        flux_per_SA_rfs.append(mJyperasec2)
    maxval = np.max(np.array(flux_per_SA_rfs))
    idx = np.where(flux_per_SA >= maxval)[0]
    return( idx )

## error propagation functions
def add_sub_error( eX, eY ):
    result = np.sqrt( np.power( eX, 2. ) + np.power( eY, 2. ) )
    return( result )

def mult_error( X, Y, eX, eY ):
    t1 = np.power( eX/X, 2. )
    t2 = np.power( eY/Y, 2. )
    result = np.sqrt( t1 + t2 ) * ( X * Y )
    return( result )

def div_error( X, Y, eX, eY ):
    t1 = np.power( eX/X, 2. )
    t2 = np.power( eY/Y, 2. )
    result = np.sqrt( t1 + t2 ) * ( X / Y )
    return( result )

## convenience functions to find AGN
def find_donley( xvals, yvals ):
    idx1 = np.where( yvals >= 0.15 )[0]
    idx2 = np.where( xvals >= 0.08 )[0]
    idx3 = np.where( yvals <= 1.21*xvals+0.27 )[0]
    idx4 = np.where( yvals >= 1.21*xvals-0.27 )[0]
    tmp_idx1 = np.intersect1d( idx1, idx2 )
    tmp_idx2 = np.intersect1d( tmp_idx1, idx3 )
    final_idx = np.intersect1d( tmp_idx2, idx4 )
    return( final_idx )

def find_lacy( xvals, yvals ):
    idx1 = np.where( xvals >= -0.1 )[0]
    idx2 = np.where( yvals >= -0.2 )[0]
    idx3 = np.where( yvals <= 0.86*xvals+0.43)[0]
    tmp_idx = np.intersect1d( idx1, idx2 )
    final_idx = np.intersect1d( tmp_idx, idx3 )
    return( final_idx )

def find_stern( xvals, yvals ):
    idx1 = np.where( xvals >= -0.01 )[0]
    idx2 = np.where( yvals >= 0.2*xvals-0.058 )[0]
    idx3 = np.where( yvals >= 2.52*xvals-0.96 )[0]
    tmp_idx = np.intersect1d( idx1, idx2 )
    final_idx = np.intersect1d( tmp_idx, idx3 )
    return( final_idx )

## convenience function to combine indices when filtering catalogue
def redo_indices( new_idx, Oagn, IRagn, Xagn, sfg, rq, lerg, herg, uncl, excess, peak, total ):

    n_Oagn = np.intersect1d( new_idx, Oagn )
    n_IRagn = np.intersect1d( new_idx, IRagn )
    n_Xagn = np.intersect1d( new_idx, Xagn )
    n_sfg = np.intersect1d( new_idx, sfg )
    n_rq = np.intersect1d( new_idx, rq )
    n_lerg = np.intersect1d( new_idx, lerg )
    n_herg = np.intersect1d( new_idx, herg )
    n_uncl = np.intersect1d( new_idx, uncl )
    n_excess = np.intersect1d( new_idx, excess )
    n_peak = np.intersect1d( new_idx, peak )
    n_total = np.intersect1d( new_idx, total )

    n_sfg_p = np.intersect1d( n_sfg, n_peak )
    n_sfg_t = np.intersect1d( n_sfg, n_total )
    n_rq_p = np.intersect1d( n_rq, n_peak )
    n_rq_t = np.intersect1d( n_rq, n_total )
    n_lerg_p = np.intersect1d( n_lerg, n_peak )
    n_lerg_t = np.intersect1d( n_lerg, n_total )
    n_herg_p = np.intersect1d( n_herg, n_peak )
    n_herg_t = np.intersect1d( n_herg, n_total )
    n_uncl_p = np.intersect1d( n_uncl, n_peak )
    n_uncl_t = np.intersect1d( n_uncl, n_total )
    n_excess_p = np.intersect1d( n_excess, n_peak )
    n_excess_t = np.intersect1d( n_excess, n_total )

    return( n_Oagn, n_IRagn, n_Xagn, n_sfg, n_rq, n_lerg, n_herg, n_uncl, n_excess, n_peak, n_total, n_sfg_p, n_sfg_t, n_rq_p, n_rq_t, n_lerg_p, n_lerg_t, n_herg_p, n_herg_t, n_uncl_p, n_uncl_t, n_excess_p, n_excess_t )

######################################################################

def get_local_rms( ra, dec, boxsize, image='' ):
    hdul = fits.open( image )
    rms_wcs = WCS( hdul[0].header )    
    x, y = rms_wcs.all_world2pix( ra, dec, 1 )
    ## Tested on an image that y is the first axis
    rms_box = hdul[0].data[int(y-boxsize/2):int(y+boxsize/2),int(x-boxsize/2):int(x+boxsize/2)].flatten()
    local_rms = np.sqrt( np.sum(np.power( rms_box[np.where(np.isfinite(rms_box))], 2. )) / len(rms_box) )
    hdul.close()
    return( local_rms )

def get_local_rms_pixel( ra, dec, image='' ):
    hdul = fits.open( image )
    rms_wcs = WCS( hdul[0].header )    
    x, y = rms_wcs.all_world2pix( ra, dec, 1 )
    ## Tested on an image that y is the first axis
    try:
        rms_val = hdul[0].data[int(y),int(x)]
    except:
        rms_val = np.nan
    hdul.close()
    return( rms_val )

def get_det_fractions( bins, parent, child ):
    det_frac = []
    e_det_frac = []
    for i in np.arange(0,len(bins)-1):
        try: 
            ndet = len(np.where( np.logical_and( child.filled(-99) >= bins[i], child.filled(-99) <= bins[i+1])  )[0] )
            ntot = len(np.where( np.logical_and( parent.filled(-99) >= bins[i], parent.filled(-99) <= bins[i+1]) )[0] )
        except:
            ndet = len(np.where( np.logical_and( child >= bins[i], child <= bins[i+1] ) )[0] )
            ntot = len(np.where( np.logical_and( parent >= bins[i], parent <= bins[i+1] ) )[0] )
        if np.logical_and( ndet > 0, ntot > 0 ):
            det_frac.append(ndet/ntot)
            e_det_frac.append(div_error(ndet,ntot,np.sqrt(ndet),np.sqrt(ntot)))
        else:
            det_frac.append(np.nan)
            e_det_frac.append(np.nan)
    return( np.asarray(det_frac), np.asarray(e_det_frac) )

def get_cum_det_fractions_rev( bins, parent, child ):
    ndet_vec = []
    ntot_vec = []
    ## reverse the bins?
#    bins = bins[::-1]
    for i in np.arange(0,len(bins)-1):
        try: 
            ndet = len(np.where( child.filled(-99) >= bins[i] )[0] )
            ntot = len(np.where( parent.filled(-99) >= bins[i] )[0] )
        except:
            ndet = len(np.where( child >= bins[i] )[0] )
            ntot = len(np.where( parent >= bins[i] )[0] )
        ndet_vec.append(ndet)
        ntot_vec.append(ntot)
    ndet_vec = np.asarray(ndet_vec)
    ntot_vec = np.asarray(ntot_vec)
    cum_frac = ndet_vec / ntot_vec
    e_cum_frac = div_error( ndet_vec, ntot_vec, np.sqrt(ndet_vec), np.sqrt(ntot_vec) )
    #return( cum_frac[::-1], e_cum_frac[::-1] )
    return( cum_frac[::-1], e_cum_frac[::-1] )


def get_cum_det_fractions( bins, parent, child, select_idx ):
    ndet_vec = []
    ntot_vec = []
    for i in np.arange(0,len(bins)-1):
        try: 
            ndet = len(np.where( np.logical_and( child.filled(-99) >= bins[i], child.filled(-99) <= bins[i+1])  )[0] )
            ntot = len(np.where( np.logical_and( parent.filled(-99) >= bins[i], parent.filled(-99) <= bins[i+1]) )[0] )
        except:
            ndet = len(np.where( np.logical_and( child >= bins[i], child <= bins[i+1] ) )[0] )
            ntot = len(np.where( np.logical_and( parent >= bins[i], parent <= bins[i+1] ) )[0] )
        ndet_vec.append(ndet)
        ntot_vec.append(ntot)
    ndet_vec = np.asarray(ndet_vec)
    ntot_vec = np.asarray(ntot_vec)
    ndet_vec = ndet_vec[select_idx]
    ntot_vec = ntot_vec[select_idx]
    det_cum = np.cumsum( ndet_vec )
    tot_cum = np.cumsum( ntot_vec )
    cum_frac = det_cum / tot_cum
    tmp = div_error( ndet_vec, ntot_vec, np.sqrt(ndet_vec), np.sqrt(ntot_vec) )
    e_cum_frac = np.sqrt( np.cumsum( np.power( tmp, 2. ) ) )
    return( cum_frac, e_cum_frac )

