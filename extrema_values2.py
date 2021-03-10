# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:05:30 2021

@author: Prosper Abega
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:59:23 2021

@author: Prosper Abega
"""
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
# This makes plots prettier
import seaborn; seaborn.set()
from scipy.signal import argrelextrema
from scipy.signal import argrelmin
from scipy.signal import argrelmax
import numpy as np
import datetime
from scipy.stats import ks_2samp
import yfinance 
import seaborn as sns


def sort_dic(d):
    s=sorted(d.keys())
    
    return {k:d[k] for k in s}



def extrema_percentage(data, comparator, percentage_extrema=0.01,percentage_rebound=1,order_left_min=1,order_left_max=1,order_right_min=1,order_right_max=1,axis=0, mode='clip'):

    if((int(order_left_min) != order_left_min) or (order_left_min < 1)):
        raise ValueError('Order left min must be an int >= 1')
    
    if((int(order_left_max) != order_left_max) or (order_left_max < order_left_min)):
        raise ValueError('Order left max must be an int >= order_left_min')
    
    if((int(order_right_min) != order_right_min) or (order_right_min < 1)):
        raise ValueError('Order right max must be an int >= 1')
    
    if((int(order_right_max) != order_right_max) or (order_right_max < order_right_min)):
        raise ValueError('Order right max must be an int >= order_right_min ')
        
        
    
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    extrema_left = np.ones(data.shape, dtype=bool)
    valid_extrema_left=np.zeros(data.shape, dtype=bool)
    
    left_borders={}
    
    main = data.take(locs, axis=axis, mode=mode)
    
    #Fist get all points that are smaller/greater  
    #than all the points on their left untill order_left_min
    for shift_left in range(1, order_left_min+1 ):
        minus = data.take(locs - shift_left, axis=axis, mode=mode)
        extrema_left &= comparator(main, minus)
        
    #Then check wether the variation (in %) is bigger than percentage_extrema
    for shift_left in range(order_left_min+1, order_left_max + 1):
        minus = data.take(locs - shift_left, axis=axis, mode=mode)
        extrema_left &= comparator(main, minus)
        valid_depth_left=np.abs(main-minus)/minus >= percentage_extrema

        valid_extrema_left_shift= np.logical_and(extrema_left,valid_depth_left)
        
        #Keep track of the left border of the extremum
        valid_for_shift=np.nonzero(valid_extrema_left_shift)[0]
        shift_left_array=np.array((shift_left,)*len(valid_for_shift))
        left_borders_values=(valid_for_shift-shift_left_array)
        left_borders_values= left_borders_values.clip(min=0)
        
        for k,el in enumerate(valid_for_shift):
        
            try :
                if np.abs(main[left_borders_values[k]]-main[el])<np.abs(main[left_borders[el]]-main[el]):
                    left_borders_values[k]=left_borders[el]
            except:
                pass
        left_borders.update(dict(zip(valid_for_shift, left_borders_values )))
        
        
        #Keep the points that satisfy both conditions
        valid_extrema_left=np.logical_or(valid_extrema_left,valid_extrema_left_shift)
        
        
    valid_extrema_step1=np.nonzero(valid_extrema_left)[0]
    valid_extrema_step1_borders=np.array([left_borders[k] for k in valid_extrema_step1])
      
    #Now we will verify the right condition
    #1) That the point is an extrema regarding the points on its right
    #2) That it keeps growing/decreasing after order_right_min
    
    try:
        extrema_left_borders = data.take(valid_extrema_step1_borders, axis=axis, mode=mode)
        extrema_step1=data.take(valid_extrema_step1, axis=axis, mode=mode)
    except:
        print('valid_extrema_step1_borders may be empty, check that the conditions are feasible')
    extrema_step1=data.take(valid_extrema_step1, axis=axis, mode=mode)
    
    
    extrema_right = np.ones(extrema_left_borders.shape, dtype=bool)
    
    
    #Fist get all points that are smaller/greater  
    #than all the points on their left untill order_left_min
    for shift_right in range(1, order_right_min+1):
        minus = data.take(valid_extrema_step1 + shift_right, axis=axis, mode=mode)
        extrema_right &= comparator(extrema_step1, minus)
    
    
    extrema_point_and_pct_change=valid_extrema_step1[extrema_right]
    valid_just_extrema = np.zeros(data.shape, dtype=bool)
    valid_just_extrema[extrema_point_and_pct_change]=True
    
    
    values_at_extrema=data.take(extrema_point_and_pct_change, axis=axis, mode=mode) 
    values_at_order_right_min=data.take(extrema_point_and_pct_change+order_right_min, axis=axis, mode=mode)  
    values_at_order_right_max=data.take(extrema_point_and_pct_change+order_right_max, axis=axis, mode=mode)
    
    # We will calculate the mean price between order right min and order right max 
    # We will label as rebound those whose mean is above order right mean 
    # this way we will avoid noisy points
    
    mean_price_after=np.zeros(len(extrema_point_and_pct_change))
    window_length=order_right_max-order_right_min
    for shift_right in range(order_right_min+1,order_right_max+1):
        mean_price_after+=data.take(extrema_point_and_pct_change+shift_right, axis=axis, mode=mode)/window_length
        
        
    rebound=(mean_price_after-values_at_order_right_min)/values_at_order_right_min
    
    if comparator==np.less:
        compared_to_order_right_min=np.greater(rebound,percentage_rebound*percentage_extrema)
    else:
        compared_to_order_right_min=np.less(rebound,-percentage_rebound*percentage_extrema)
    
    
    is_rebound = compared_to_order_right_min
    
    valid_rebound = np.zeros(data.shape, dtype=bool)
    valid_rebound[extrema_point_and_pct_change[is_rebound]]=True
    
    # Finally we keep the rigt and left point verifing the condition
    final_valid_points=np.logical_and(valid_extrema_left,valid_rebound)
    
    
    d_cleaned={k: left_borders[k] for k in left_borders.keys() if valid_just_extrema[k]}
    left_borders=d_cleaned
    extrema_left_borders = data.take(np.array([*left_borders.values()]), axis=axis, mode=mode)
    
    return valid_just_extrema,valid_rebound,sort_dic(left_borders),extrema_left_borders


def get_extrema(data,percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max):

    s=data.values
    
    res_mins=extrema_percentage(s, np.less, percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max)
    
    res_maxs=extrema_percentage(s, np.greater, percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max)
    
    minimas_info=list(res_mins)
    maximas_info=list(res_maxs)
    
    return minimas_info,maximas_info

def create_feautures(df,window_sizes_X,importance_left,importance_right):
    for rol in window_sizes_X:
        
        if rol<=importance_left:
            #df['std_before_'+str(rol)]=df['return'].rolling(rol).std()
            
            df['ret_before_'+str(rol)]=df['return'].rolling(rol).sum()
            
            df['volume_diff_before_'+str(rol)]=df['Volume_incr'].rolling(rol).sum()
            
            #df['volume_diff_std_before_'+str(rol)]=df['Volume_incr'].rolling(rol).std()
            
            df['biggest_spread_day_before_'+str(rol)]=df['Biggest_spread_day'].rolling(rol).mean()
            
            df['spread_day_before_'+str(rol)]=df['Spread_day'].rolling(rol).mean()
            
            df['spread_overnight_before_'+str(rol)]=df['Spread_overnight'].rolling(rol).mean()
            
            df['Return_per_Volume_before_'+str(rol)]=df['Return_per_Volume'].shift(rol).replace([np.inf, -np.inf], np.nan)
            
            df['Return_per_Volume_mean_before_'+str(rol)]=df['Return_per_Volume'].replace([np.inf, -np.inf], np.nan).shift(rol)
            

        if rol<importance_right:
         
            #df['std_after_'+str(rol)]=df['return'].rolling(rol).std().shift(-rol)

            df['ret_after_'+str(rol)]=df['return'].rolling(rol).sum().shift(-rol)

            df['volume_diff_after_'+str(rol)]=df['Volume_incr'].rolling(rol).sum().shift(-rol)

            #df['volume_diff_std_after_'+str(rol)]=df['Volume_incr'].rolling(rol).std().shift(-rol)

            df['biggest_spread_day_after_'+str(rol)]=df['Biggest_spread_day'].rolling(rol).mean().shift(-rol)

            df['spread_day_after_'+str(rol)]=df['Spread_day'].rolling(rol).mean().shift(-rol)

            df['spread_overnight_after_'+str(rol)]=df['Spread_overnight'].rolling(rol).mean().shift(-rol)
            
            df['Return_per_Volume_after_'+str(rol)]=df['Return_per_Volume'].shift(-rol).replace([np.inf, -np.inf], np.nan)


def get_model_stock(percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max,stock,start,end,window_sizes_X):
    
    df =  yfinance.download(stock, start=start, end=end)
    
        
    df['Ticker']=[stock]*len(df)
    df['price']=df['Adj Close']
    df['return']=np.log(df.price) - np.log(df.price.shift(1))
    df['Volume_incr']=np.log(df['Volume'])-np.log(df['Volume'].shift(1))
    df['Biggest_spread_day']=(df['High']-df['Low'])/df['Low']
    df['Spread_day']=(df['Close']-df['Open'])/df['Open']
    df['Spread_overnight']=(df['Open']-df['Close'].shift(1))/df['Close'].shift(1)   
    df['Return_per_Volume']=df['return']/df['Volume_incr']
    
    minimas_info,maximas_info=get_extrema(df.price,percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max)
      
    create_feautures(df,window_sizes_X,order_left_max,order_right_min)
    
    df_mins=df.iloc[minimas_info[0]]
    width_left=[ k-minimas_info[2][k] for k in minimas_info[2].keys()]
    df_mins['width_left']=width_left
    df_mins['return_left_borders_per_day']=(np.log(df_mins.price.values)-np.log(minimas_info[3]))/width_left

    df_maxs=df.iloc[maximas_info[0]]
    width_left=[ k-maximas_info[2][k] for k in maximas_info[2].keys()]
    df_maxs['width_left']=width_left
    df_maxs['return_left_borders_per_day']=(np.log(df_maxs.price.values)-np.log(maximas_info[3]))/width_left
    
    df_mins['is_rebound']=1*minimas_info[1][minimas_info[0]]
    df_maxs['is_rebound']=1*maximas_info[1][maximas_info[0]]
    
    return df_mins,df_maxs


def get_model_intraday(percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max,window_sizes_X):
    #df=pd.read_csv('MSFT_200101_200701.csv',sep=";")
    df=pd.read_csv('MSFT_190101_200101.csv',sep=",")
    
    cols={'<TICKER>':'Ticker',"<PER>":'Per',"<DATE>":'Date',"<TIME>":'Time',"<OPEN>":'Open',"<HIGH>":'High',"<LOW>":'Low',"<CLOSE>":'Close',"<VOL>":'Volume'}
    df.rename(columns=cols,inplace=True)
    
    df['price']=df['Close']
    df['return']=np.log(df.price) - np.log(df.price.shift(1))
    df['Volume_incr']=np.log(df['Volume'])-np.log(df['Volume'].shift(1))
    df['Biggest_spread_day']=(df['High']-df['Low'])/df['Low']
    df['Spread_day']=(df['Close']-df['Open'])/df['Open']
    df['Spread_overnight']=(df['Open']-df['Close'].shift(1))/df['Close'].shift(1)   
    df['Return_per_Volume']=df['return']/df['Volume_incr']

    minimas_info,maximas_info=get_extrema(df.price,percentage_extrema,percentage_rebound,order_left_min,order_left_max,order_right_min,order_right_max)

    create_feautures(df,window_sizes_X,order_left_max,order_right_min)

    df_mins=df.iloc[minimas_info[0]]
    width_left=[ k-minimas_info[2][k] for k in minimas_info[2].keys()]
    df_mins['width_left']=width_left
    df_mins['return_left_borders_per_day']=(np.log(df_mins.price.values)-np.log(minimas_info[3]))/width_left

    df_maxs=df.iloc[maximas_info[0]]
    width_left=[ k-maximas_info[2][k] for k in maximas_info[2].keys()]
    df_maxs['width_left']=width_left
    df_maxs['return_left_borders_per_day']=(np.log(df_maxs.price.values)-np.log(maximas_info[3]))/width_left
    
    df_mins['is_rebound']=1*minimas_info[1][minimas_info[0]]
    df_maxs['is_rebound']=1*maximas_info[1][maximas_info[0]]


    return df_mins,df_maxs,df

def get_all_kolmogorov_tests(df_mins,df_maxs):

        

    p_values_mins=pd.Series()
    p_values_maxs=pd.Series()
    feature_columns=[col for col in df_mins.columns]
    for col in feature_columns:
        p_values_mins.loc[col]=ks_2samp(df_mins[df_mins['is_rebound']==1][col],df_mins[df_mins['is_rebound']==0][col])[1]
        p_values_maxs.loc[col]=ks_2samp(df_maxs[df_maxs['is_rebound']==1][col],df_maxs[df_maxs['is_rebound']==0][col])[1]

    return p_values_mins,p_values_maxs


def get_disciminant_features(confidence_level,p_vals):
    
    nb_most_significant=(p_vals<confidence_level).sum()
    most_significant=p_vals.nsmallest(nb_most_significant).index
    print('most significant',p_vals.nsmallest(nb_most_significant))
    
    return nb_most_significant,most_significant

def visualisation(nb_most_significant,most_significant,df):
    fig,axes=plt.subplots(nrows=1 + nb_most_significant//3,ncols=3,figsize=(14,14))
    
    df1=df[df['is_rebound']==0]
    df2=df[df['is_rebound']==1]
    
    for i,col in enumerate(most_significant):
        
        sns.distplot(df1[col],color='blue',ax=fig.axes[i])
        sns.distplot(df2[col],color='red',ax=fig.axes[i])

    plt.tight_layout()