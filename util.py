import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle


def preprocessing_simple_train(csvfile='rainfall_train.csv'):
    train_df = pd.read_csv('rainfall_train.csv')
    train_df = train_df[['rainfall_train.stn4contest', 'rainfall_train.dh',
       'rainfall_train.ef_year', 'rainfall_train.ef_month',
       'rainfall_train.ef_day', 'rainfall_train.ef_hour', 'rainfall_train.v01',
       'rainfall_train.v02', 'rainfall_train.v03', 'rainfall_train.v04',
       'rainfall_train.v05', 'rainfall_train.v06', 'rainfall_train.v07',
       'rainfall_train.v08', 'rainfall_train.v09', 'rainfall_train.vv',
       'rainfall_train.class_interval']]
    tmp = train_df.groupby(by=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])['rainfall_train.dh'].min().reset_index()
    result = pd.merge(train_df,tmp, on=['rainfall_train.stn4contest','rainfall_train.ef_year','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour'])
    result = result[result['rainfall_train.dh_x'] == result['rainfall_train.dh_y']].drop(columns=['rainfall_train.dh_x','rainfall_train.dh_y'])
    for i in range(1,9):
        result[f'rainfall_train.v0{i}'] -=  result[f'rainfall_train.v0{i+1}']
        result[f'rainfall_train.v0{i}'] /= 100.0
    result[f'rainfall_train.v09'] /= 100.0
    return result


def preprocessing_simple_test(csvfile='rainfall_test.csv'):
    df = pd.read_csv('rainfall_test.csv')
    df = df.drop(columns=['Unnamed: 0', 'rainfall_test.fc_year', 'rainfall_test.fc_month',
       'rainfall_test.fc_day', 'rainfall_test.fc_hour','rainfall_test.ef_year'])
    tmp = df.groupby(by=['rainfall_test.stn4contest','rainfall_test.ef_month','rainfall_test.ef_day','rainfall_test.ef_hour'])['rainfall_test.dh'].min().reset_index()
    tmp2 = pd.merge(df,tmp,on=['rainfall_test.stn4contest','rainfall_test.ef_month','rainfall_test.ef_day','rainfall_test.ef_hour'])
    result = tmp2[tmp2['rainfall_test.dh_x']==tmp2['rainfall_test.dh_y']].drop(columns=['rainfall_test.dh_x','rainfall_test.dh_y'])
    result.columns = np.array(['STN', 'DHX', 'M', 'D','H', 'V1', 'V2','V3', 'V4', 'V5','V6', 'V7', 'V8','V9', 'class_interval'])
    for i in range(1,9):
        result[f"V{i}"] = result[f"V{i}"] -result[f"V{i+1}"]
    for i in range(1,10):
        result[f"V{i}"] = result[f"V{i}"] / 100.0
    return result


def preprocessing_daegun(csvfile='rainfall_train.csv'):
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

def preprocessing_ML_daegun(csvfile='rainfall_train.csv'):
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