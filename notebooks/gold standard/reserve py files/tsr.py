import numpy as np
from numpy import array
import math
from collections import namedtuple
from tsr import linear
from fox_toolbox.utils import volatility

"LINEAR TSR"

def minmax_strikes(vol, expiry, fwd, nb):
    """
    NORMAL VOL
    
    vol      rates.Volatility
    expiry
    fwd
    nb
    
    'caplet strikes': (fwd, Kmax),
    'floorlet strikes': (Kmin, fwd) 
    """
    
    n_inv = 5.730390
    vol_atm = vol.value
    std = n_inv*vol_atm*np.sqrt(expiry)
    
    if vol.type == 'N':
        Smin = fwd - std
        Smax = fwd + std
        kstep = (Smax - Smin)/nb
        kmin = Smin + kstep
        kmax = Smax - kstep
    else:
        NotImplemented('vol type other that N is not yer implemented')
    
    minmax_strikes = namedtuple('minmax_strikes_nvol', 'kmin kmax kstep fwd volType')
    return minmax_strikes(kmin, kmax, kstep, fwd, vol.type)

def build_strike_ladders(minmax_strikes, neff_capl, neff_floo):
    if minmax_strikes.volType == 'N':
        caplet_strikes = np.linspace(minmax_strikes.fwd, minmax_strikes.kmax, neff_capl)
        floorlet_strikes = np.linspace(minmax_strikes.kmin, minmax_strikes.fwd, neff_floo)
    else:
        NotImplemented('vol type other that N is not yer implemented')
    
    strike_ladders = namedtuple('strike_ladders', 'floorlet_ladder caplet_ladder')
    return strike_ladders(floorlet_strikes, caplet_strikes)
    
def build_weights(minmax_strikes, neff_capl, neff_floo, tsr_coeff):
    w0c = tsr_coeff.a * (minmax_strikes.fwd + minmax_strikes.kstep) + tsr_coeff.b
    wic = [2 * tsr_coeff.a * minmax_strikes.kstep for _ in range(neff_capl - 1)]
    
    wif = [-2 * tsr_coeff.a * minmax_strikes.kstep for _ in range(neff_floo - 1)]
    w0f = tsr_coeff.a * (minmax_strikes.fwd - minmax_strikes.kstep) + tsr_coeff.b
    
    tsr_weights = namedtuple('tsr_weights', 'capletWeights floorletWeights')
    return tsr_weights(array([w0c] + wic), array(wif + [w0f]))
    
def get_DiscOverAnnuity(strikes, tsr_coeff):
    return tsr_coeff.a * strikes + tsr_coeff.b
    
def get_neff(n):
    return n


def tsr_model(swo, dsc_curve, estim_curve, n, mr, payment_date):
    """
    swo - rates.Swaption with single or smile volatility

    Model settings:
    n - number of replication strikes
    mr - mean reversion
    """
    neff = get_neff(n)
    fwd = swo.get_swap_rate(dsc_curve, estim_curve)
    tsr_strikes = minmax_strikes(swo.vol, swo.expiry, fwd, neff)
    
    neff_capl = math.ceil((tsr_strikes.kmax - tsr_strikes.fwd)/tsr_strikes.kstep) + 1
    neff_floo = math.floor((tsr_strikes.fwd - tsr_strikes.kmin)/tsr_strikes.kstep) + 1
    
    
    strikes_ladders = build_strike_ladders(tsr_strikes, neff_capl, neff_floo)
    tsr_coeff = linear.get_coeff(payment_date, dsc_curve, swo, mr, estim_curve)
    tsr_weights = build_weights(tsr_strikes, neff_capl, neff_floo, tsr_coeff)
    
    myBachelierCaplet = array([volatility.BachelierPrice(F=swo.get_swap_rate(dsc_curve, estim_curve), K=strike, v=swo.vol.value*np.sqrt(swo.expiry)) 
    * swo.get_annuity(dsc_curve) / dsc_curve.get_dsc(swo.start_date) for strike in strikes_ladders.caplet_ladder])
    
    myBachelierFloorlet = array([volatility.BachelierPrice(F=swo.get_swap_rate(dsc_curve, estim_curve), K=strike, v=swo.vol.value*np.sqrt(swo.expiry), w=-1) 
    * swo.get_annuity(dsc_curve) / dsc_curve.get_dsc(swo.start_date) for strike in strikes_ladders.floorlet_ladder])
    
    cms_caplet = tsr_weights.capletWeights.dot(myBachelierCaplet)
    cms_floorlet = tsr_weights.floorletWeights.dot(myBachelierFloorlet)
    
    cms_swaplet = cms_caplet - cms_floorlet + swo.strike * dsc_curve.get_fwd_dsc(swo.expiry, payment_date)
    
    return cms_swaplet
    