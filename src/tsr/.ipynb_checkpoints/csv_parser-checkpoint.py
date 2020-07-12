import numpy as np
import pandas as pd
from fox_toolbox.utils.rates import Curve, RateCurve, Swap, Swaption, Volatility
from collections import namedtuple

swap_rate_model = namedtuple('swap_rate_model', 'mtype a b neff')
cms_result = namedtuple('cms_result', 'swap_fwd disc_Tf_Tp')
csvCMSFlow = namedtuple('csvCMSFlow', 'CMS_length vol fixing_date pmnt_date strike model n strike_min strike_max callput calib_basket fwd discTfTp')
csvSwaplet = namedtuple('csvSwaplet', 'swaplet disc_Tf_Tp strike caplet floorlet adjCMSrate')
pricingCMSflow = namedtuple('pricingCMSflow', 'caplet floorlet swaplet')

def get_calib_basket(one_column_df):
    tsr_columns = ['Index','Strike','RepFwd','Weights','Disc/A','Vol','AdjVol', 'SwoPrice', 'AdjSwoPrice', 'Vega']
    
    cal_basket = pd.DataFrame(columns=tsr_columns)
    for _, csv_line in one_column_df.iterrows():
        pars = np.array(csv_line[0].split(' ; ')[:-1])
        good = pd.Series(pars, index=tsr_columns, dtype=float)
        cal_basket = cal_basket.append(good, ignore_index=True)
        
    return cal_basket

def prase_cms_flow(df):
    
    df = df[3:].reset_index(drop=True)
    
    line_5 = df.loc[0,0].replace(';', '')
    assert line_5 in ['CMS Caplet', 'CMS Floorlet'], '5th line in cms csv log is not CMS Capelt nor CMS Floorlet'
    cms_dict = {}
    model = str(df.loc[6,0].split(';')[1]).replace(" ", "")
    
    if model == 'Linear':
        parse_idxs = [1, 2, 3, 8, 9, 11, 12, 13, 14, 16]
        call_calib_idx =16
    elif model == 'Hagan':
        parse_idxs = [1, 2, 3, 8, 9, 10, 11]
        call_calib_idx = 12
    else:
        ValueError('model is not Linear nor Hagan')
    
    for i in parse_idxs:
        str_key   = df.loc[i,0].split(';')[0]
        val = float(df.loc[i,0].split(';')[1])
        cms_dict[str_key] = val
    
    for str_key, float_val in zip(df.loc[4,0].split(';'), df.loc[5,0].split(';')[:-1]):
        cms_dict[str_key] = float(float_val)
    
    if model == 'Linear': 
        l = int(cms_dict['Effective number of replication points '])
        excel_step = 16 + l
        swap_rate_model_ = swap_rate_model(mtype=model, a=cms_dict['a '], b=cms_dict['b '], neff=l)
    elif model == 'Hagan':
        l = int(cms_dict['User number of replication points '])
        excel_step = 12 + l
        swap_rate_model_ = model
        
    one_column_df = df[call_calib_idx:call_calib_idx + l]
    cal_basket = get_calib_basket(one_column_df)
    
    instrument = csvCMSFlow(CMS_length=cms_dict['CMS length'],
                        vol=None,
                        fixing_date=cms_dict['Fixing date Tf'],
                        pmnt_date=cms_dict['Payment date Tp'],
                        strike=cms_dict['CMS strike'],
                        model=swap_rate_model_,
                        n=int(cms_dict['User number of replication points ']),
                        strike_min=cms_dict['Strike min'],
                        strike_max=cms_dict['Strike max'],   
                        callput=line_5,
                        calib_basket=cal_basket,
                        fwd = cms_dict['Swaption Forward'],
                        discTfTp=cms_dict['Discount factor between Tf and Tp'])
    
    return instrument, excel_step


def parse_cms_swaplet(df):
    parse_idxs = np.array([1,2,3,4,5,9]) + 3
    cms_dict = {}
    for i in parse_idxs:
        str_key   = df.loc[i,0].split(';')[0]
        val = float(df.loc[i,0].split(';')[1].strip())
        cms_dict[str_key] = val
    return csvSwaplet(swaplet = cms_dict['CMS swaplet'],
                      disc_Tf_Tp = cms_dict['Discount factor between Tf and Tp'],
                      strike = cms_dict['CMS strike'],
                      caplet = cms_dict['CMS caplet'],
                      floorlet = cms_dict['CMS floorlet'],
                      adjCMSrate = cms_dict['Adjusted CMS rate'],
                     )
    

def parse_pricingCMSflow(df):
    cms_caplet, l1 = prase_cms_flow(df)
    cms_floorlet, l2 = prase_cms_flow(df[l1:].reset_index(drop=True))
    cms_swaplet = parse_cms_swaplet(df[l1+l2:].reset_index(drop=True))
    
    return cms_caplet, cms_floorlet, cms_swaplet, l1+l2

def parse_pricingCMScap(df):
    pass
    
    
def parse_cmsflow(df):
    rest_idx = 19
    
    caplet, floorlet, swaplet, capfindex = parse_pricingCMSflow(df)
    capflos = [pricingCMSflow(caplet, floorlet, swaplet)]
    cfx = capfindex + rest_idx
    
    while not df[cfx:].empty:
        caplet, floorlet, swaplet, capfindex = parse_pricingCMSflow(df[cfx:].reset_index(drop=True))
        cfx += capfindex + rest_idx
        capflos.append(pricingCMSflow(caplet, floorlet, swaplet))
    
    return  list(reversed(capflos))

def parse_cmscap(csv):
    pass
    
def parse_csv(csv):
    df = pd.read_csv(csv, error_bad_lines=False, header = None, warn_bad_lines=False)
    product = df.iloc[1,0].split('()')[0].split()[-1]
    print(f'Parsing {product} csv log')
    
    if product == 'CmsCap':
        return parse_cmscap(df)
    elif product == 'CmsFlow':
        return parse_cmsflow(df)
    else:
        raise KeyError (f'{product} should match CmsCap, CmsFlow')

