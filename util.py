import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

train_data_path = "./data/rainfall_train.csv"
test_data_path = "./data/rainfall_test.csv"


def make_mean(df, _n = 15):
    # previously make v_max, v_median column
    stn_array = df['rainfall_train.stn4contest'].unique()
    year_array = df['rainfall_train.ef_year'].unique()
    df['mean_vmax'] = None
    df['mean_vmedian'] = None
    for y in year_array:
        tmp_df = df[df['rainfall_train.ef_year'] == y]
        for _stn in stn_array:
            _array = tmp_df[tmp_df['rainfall_train.stn4contest'] == _stn].sort_values(by=['rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])[['rainfall_train.ef_year','rainfall_train.stn4contest','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour','v_max','v_median']]
            N = len(_array)
            mean_vmax = []
            mean_vmedian = []
            for idx  in range(N):
                start = max(idx-_n, 0)
                end = min(idx+_n,N-1)
                mean_vmax.append(_array['v_max'].values[start:end].mean())
                mean_vmedian.append(_array['v_median'].values[start:end].mean())
            _array['avg_vmax'] = np.array(mean_vmax)
            _array['avg_vmedian'] = np.array(mean_vmedian)
            _array = _array.drop(columns=['v_max','v_median'])
            df = pd.merge(df,_array,how='left',on=['rainfall_train.ef_year','rainfall_train.stn4contest','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])
            df['mean_vmax'] = df['avg_vmax'].combine_first(df['mean_vmax']) 
            df['mean_vmedian'] = df['avg_vmedian'].combine_first(df['mean_vmedian']) 
            df = df.drop(columns=['avg_vmax','avg_vmedian'])
    return df

def make_day2vv(df):
    #please make 'day' column
    tmp = df.groupby(by=['day'])['rainfall_train.vv'].mean().reset_index()
    tmp['rainfall_train.vv'] = StandardScaler().fit_transform(tmp[['rainfall_train.vv']])
    return tmp

def make_day2class(df):
    #please make 'day' column
    tmp = df.groupby(by=['day'])['rainfall_train.class_interval'].mean().reset_index()
    return tmp
def make_day2stdclass(df):
    #please make 'day' column
    tmp = df.groupby(by=['day'])['rainfall_train.class_interval'].std().reset_index()
    return tmp

def make_day2freqclass(df):
    df = df.copy()
    df = df[df['rainfall_train.class_interval'] !=0]
    tmp = df.groupby(by=['day'])['rainfall_train.class_interval'].apply(lambda x: np.argmax(np.bincount(np.array(x)))).reset_index()
    return tmp
def month_to_day(month):
    month_to_day = np.array([0,31,28,31,30,31,30,31,31,30,31,30,31])
    for i in range(1,13):
        month_to_day[i] += month_to_day[i-1] 
    return month_to_day[month]

def preprocessing_simple_train(_method="fast"):
    train_df = pd.read_csv(train_data_path)
    train_df = train_df[['rainfall_train.stn4contest', 'rainfall_train.dh',
       'rainfall_train.ef_year', 'rainfall_train.ef_month',
       'rainfall_train.ef_day', 'rainfall_train.ef_hour', 'rainfall_train.v01',
       'rainfall_train.v02', 'rainfall_train.v03', 'rainfall_train.v04',
       'rainfall_train.v05', 'rainfall_train.v06', 'rainfall_train.v07',
       'rainfall_train.v08', 'rainfall_train.v09', 'rainfall_train.vv',
       'rainfall_train.class_interval']]
    train_df = train_df[train_df["rainfall_train.class_interval"] != -999]
    for i in range(1,9):
        train_df[f'rainfall_train.v0{i}'] -=  train_df[f'rainfall_train.v0{i+1}']
        train_df[f'rainfall_train.v0{i}'] /= 100.0
    train_df[f'rainfall_train.v09'] /= 100.0
    if _method == "fast":
        tmp = train_df.groupby(by=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])['rainfall_train.dh'].min().reset_index()
        result = pd.merge(train_df,tmp, on=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])
        result = result[result['rainfall_train.dh_x'] == result['rainfall_train.dh_y']].drop(columns=['rainfall_train.dh_y'])
    if _method == 'mean':
        tmp = train_df.groupby(by=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour']).mean().reset_index()
        result = tmp.drop(columns=['rainfall_train.dh'])
    if _method == 'median':
        tmp = train_df.groupby(by=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour']).median().reset_index()
        result = tmp.drop(columns=['rainfall_train.dh'])
    tmp = 1 - result[['rainfall_train.v01','rainfall_train.v02','rainfall_train.v03','rainfall_train.v04','rainfall_train.v05',
        'rainfall_train.v06','rainfall_train.v07','rainfall_train.v08','rainfall_train.v09']].apply(lambda x: sum(x),axis=1)
    result['rainfall_train.v00'] = tmp
    result['v_max'] = result[['rainfall_train.v00','rainfall_train.v01','rainfall_train.v02','rainfall_train.v03','rainfall_train.v04','rainfall_train.v05',
        'rainfall_train.v06','rainfall_train.v07','rainfall_train.v08','rainfall_train.v09']].apply(lambda x: max(enumerate(x),key=lambda x: x[1])[0],axis=1)
    _median = np.array([0.15,0.35,0.75,1.5,3.5,7.5,15.0,25.0,30.0])
    result['v_median'] = result[['rainfall_train.v01','rainfall_train.v02','rainfall_train.v03','rainfall_train.v04','rainfall_train.v05',
        'rainfall_train.v06','rainfall_train.v07','rainfall_train.v08','rainfall_train.v09']].apply(lambda x: (x * _median).sum(),axis=1)
    result.drop(columns=['rainfall_train.v00'])
    return result


def preprocessing_simple_test(csvfile=test_data_path):
    df = pd.read_csv(csvfile)
    df = df.drop(columns=['Unnamed: 0', 'rainfall_test.fc_year', 'rainfall_test.fc_month',
       'rainfall_test.fc_day', 'rainfall_test.fc_hour'])
    df = df[df["rainfall_test.class_interval"] != -999]
    
    tmp = df.groupby(by=['rainfall_test.stn4contest','rainfall_test.ef_month','rainfall_test.ef_day','rainfall_test.ef_hour'])['rainfall_test.dh'].min().reset_index()
    tmp2 = pd.merge(df,tmp,on=['rainfall_test.stn4contest','rainfall_test.ef_month','rainfall_test.ef_day','rainfall_test.ef_hour'])
    result = tmp2[tmp2['rainfall_test.dh_x']==tmp2['rainfall_test.dh_y']].drop(columns=['rainfall_test.dh_y'])

    for i in range(1,9):
        result[f"rainfall_test.v0{i}"] -= result[f"rainfall_test.v0{i+1}"]
        result[f"rainfall_test.v0{i}"] /= 100.0
    result[f"rainfall_test.v09"] /= 100.0
    tmp = 1 - result[['rainfall_test.v01','rainfall_test.v02','rainfall_test.v03','rainfall_test.v04','rainfall_test.v05',
        'rainfall_test.v06','rainfall_test.v07','rainfall_test.v08','rainfall_test.v09']].apply(lambda x: sum(x),axis=1)
    result['rainfall_test.v00'] = tmp
    result['v_max'] = result[['rainfall_test.v00','rainfall_test.v01','rainfall_test.v02','rainfall_test.v03','rainfall_test.v04','rainfall_test.v05',
        'rainfall_test.v06','rainfall_test.v07','rainfall_test.v08','rainfall_test.v09']].apply(lambda x: max(enumerate(x),key=lambda x: x[1])[0],axis=1)
    _median = np.array([0.15,0.35,0.75,1.5,3.5,7.5,15.0,25.0,30.0])
    result['v_median'] = result[['rainfall_test.v01','rainfall_test.v02','rainfall_test.v03','rainfall_test.v04','rainfall_test.v05',
        'rainfall_test.v06','rainfall_test.v07','rainfall_test.v08','rainfall_test.v09']].apply(lambda x: (x * _median).sum(),axis=1)
    result.drop(columns=['rainfall_test.v00'])
    return result


def preprocessing_daegun(csvfile=train_data_path):
    rainfall_train = pd.read_csv(csvfile)
    rainfall_train.drop(columns=['Unnamed: 0'],inplace= True)

    df = pd.concat([pd.read_csv('daegun_first.csv'),rainfall_train[['rainfall_train.dh','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour','rainfall_train.ef_year']]],axis=1)
    df = df.drop(columns=['TM_FC','TM_EF','EF_class'])
    null_df = df[df['class'] == -999]
    df = df[df['class'] != -999]

    month_to_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(1,12):
        month_to_day[i] += month_to_day[i-1] 
    month_to_day = {idx+2 : i for idx, i in enumerate(month_to_day)}
    month_to_day[1] = 0
    df['day'] = df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']
    df['day_sin'] = np.sin(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df['day_cos'] = np.cos(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df =df.drop(columns=['rainfall_train.ef_month','rainfall_train.ef_day'])

    df['hour_sin'] = np.sin(2 *np.pi * df['rainfall_train.ef_hour'] /24)
    df['hour_cos'] = np.cos(2 *np.pi * df['rainfall_train.ef_hour'] /24)


    scaler = MinMaxScaler()
    scaler.fit(df[['DH']])
    df[['DH']]=scaler.transform(df[['DH']])
    df[[f"V{i}" for i in range(1,10)]] = df[[f"V{i}" for i in range(1,10)]] / 100
    return df

def preprocessing_ML_daegun(csvfile=train_data_path):
    rainfall_train = pd.read_csv(csvfile)
    rainfall_train.drop(columns=['Unnamed: 0'],inplace= True)

    df = pd.concat([pd.read_csv('daegun_first.csv'),rainfall_train[['rainfall_train.dh','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour','rainfall_train.ef_year']]],axis=1)
    df = df.drop(columns=['TM_FC','TM_EF','EF_class'])
    null_df = df[df['class'] == -999]
    df = df[df['class'] != -999]

    month_to_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(1,12):
        month_to_day[i] += month_to_day[i-1] 
    month_to_day = {idx+2 : i for idx, i in enumerate(month_to_day)}
    month_to_day[1] = 0
    df['day'] = df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']
    df['day_sin'] = np.sin(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df['day_cos'] = np.cos(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df =df.drop(columns=['rainfall_train.ef_month','rainfall_train.ef_day'])

    df['hour_sin'] = np.sin(2 *np.pi * df['rainfall_train.ef_hour'] /24)
    df['hour_cos'] = np.cos(2 *np.pi * df['rainfall_train.ef_hour'] /24)


    scaler = MinMaxScaler()
    scaler.fit(df[['DH']])
    df[['DH']]=scaler.transform(df[['DH']])
    df[[f"V{i}" for i in range(1,10)]] = df[[f"V{i}" for i in range(1,10)]] / 100

    #model
    with open('binary_classification.pkl','rb') as f:
        bmodel = pickle.load(f)
    with open('regression_random.pkl','rb') as f:
        rmodel = pickle.load(f)
    x = df.drop(columns=['STN','V0','rainfall_train.ef_year','rainfall_train.dh','day','rainfall_train.ef_hour','class','VV'])
    yr = rmodel.predict(x)
    yc = bmodel.predict(x)
    df['RP'] = yr.astype(float)
    df['BP'] = yc.astype(float)
    return df


def compute_csi(y_true,y_pred,regression=False):        
    # compute csi
    h =((y_true != 0) & (y_pred == y_true)).sum()
    f = (y_pred != y_true).sum()
    return h/(h+f)




def all_csi_score(data, binary_model, c_model):
    #data:pd.Dataframe
    #binary_model: true-> 무강수, false -> 강수
    #c_model: 
    #DF column 
    #DH, VV, V0-9, class, rainfall_train.ef_hour, day, day_sin, day_cos, hour_sin, hour_cos
    b_x = data.drop(columns=['rainfall_train.ef_hour','day','VV','class'])
    b_y = (data['class'] == 0)
    b_y_pred = binary_model.predict(b_x)
    after_df = data[~b_y_pred].sort_values(by=['day','rainfall_train.ef_hour']).reset_index(drop=True)
    l_y = after_df['class']
    l_x = after_df.drop(columns=['VV','class']).values
    l_x = l_x.reshape(l_x.shape[0],1,l_x.shape[1])
    l_y_pred = c_model.predict(l_x)
    c = (b_y_pred == b_y).sum()
    m = (b_y_pred != b_y).sum()
    h=(np.argmax(l_y_pred, axis=-1) == l_y).sum()
    f= (np.argmax(l_y_pred, axis=-1) != l_y).sum()
    return h/(h+m+f)