from hw.hw_helper import _B
import numpy as np
from numpy import array
from collections import namedtuple

tsr_ab = namedtuple('tsr_ab', 'a b gamma annuity swaprate')

def get_coeff(pmnt_date, dsc_curve, swo, mr, fwd_curve):
    """ 
    pmnt_date   float              payment date (could be any date you want)
    dsc_curve   rates.RateCurve    discount curve of type 
    swo         rates.Swaption      is used for swap schedule and computing:
                                        expirty(t)
                                        swap rate
                                        annuity
    mr          float              mean reversion in Hull White model
    fwd_curve   rates.RateCurve    define mono or multi curve framework
    
    return      float              Hull White mean reversion approximation for linear tsr parameter a.
                                   Check Piterbarg p.713 (mono curve only)
    """
    t = swo.expiry
    M = pmnt_date
    pmnt_dates = swo.payment_dates # np.round(swo.payment_dates, 6)
    
    if fwd_curve is not None:
        flt_adjs = swo.get_flt_adjustments(dsc_curve, fwd_curve)
    else:
        flt_adjs = np.zeros_like(swo.payment_dates)
        
    fwd = swo.get_swap_rate(dsc_curve, flt_adjs=flt_adjs)
    annuity = swo.get_annuity(dsc_curve)
    
    hw_Bs = _B(t, pmnt_dates, mr)
    gamma = swo.get_annuity_product(dsc_curve, hw_Bs)/annuity
    
    num = dsc_curve.get_dsc(M) * (gamma - _B(t,M, mr))
    denum = dsc_curve.get_dsc(swo.maturity) * _B(t, swo.maturity, mr) +\
            - dsc_curve.get_dsc(swo.start_date) * _B(t, swo.start_date, mr) +\
            fwd * gamma * annuity
    
    #multi curve adjustment
    bi = _B(t, pmnt_dates, mr)
    multi_adj = - swo.get_annuity_product(dsc_curve, bi * flt_adjs)
    
    a = num/(denum + multi_adj)
    b = dsc_curve.get_dsc(M)/annuity - a * fwd
    
    return tsr_ab(a, b, gamma, annuity, fwd)

    

    
    