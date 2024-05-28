from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor
import talib


def end_model(timestep=5):
    input_X = keras.layers.Input((timestep, None,16))
    X = keras.LSTM(50)(input_X)
    # timestep, 50
    outputs, h, c= keras.layers.LSTM(10,return_state=True)()
    # outputs (10)
    # input_D = keras.layers.Input()(outputs)
    encode_state = [h,c]
    #decode
    keras.layer.LSTM(10, return_squences=True, return_state=True, intial_state=encode_state)()

    return keras.Model(inputs=input_X, outputs=X)

def dstm_model(timestep=5):
    input_X = keras.layers.Input((timestep,20,16))
    input_X1 = input_X[:,:,:,0:14]
    input_X2 = input_X[:,-1,-1,14]
    input_X3 = input_X[:,-1,-1,15]
    # None, timestep, 20, 14
    channel_process = []
    for i in range(timestep):
        channel_ = input_X1[:,i,:,:] 
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,16))(channel_)
        channel_ = keras.layers.Dropout(0.2)(channel_)
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        channel_ = keras.layers.BatchNormalization()(channel_)
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        channel_ = keras.layers.Dropout(0.2)(channel_)
        channel_=keras.layers.LSTM(10, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        channel_ = keras.layers.BatchNormalization()(channel_)
        #output = None,1,10
        channel_process.append(channel_)
    X1 = keras.layers.Reshape((1,))(input_X2)
    X2 =keras.layers.Reshape((1,))(input_X3)
 
    #(timestep,)
    X = keras.layers.Concatenate()(channel_process)
    X = keras.layers.Reshape((timestep,10))(X)
    # None, timestep, 10
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,10))(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,50))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,50))(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(50,recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,50))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(30, activation="relu")(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(15, activation="relu")(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.Dense(10, activation="relu")(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Concatenate(axis=-1)([X,X2])
    X = keras.layers.Dense(10, activation="softmax")(X)
    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel

def fstm_model(timestep=5):
    #(None,Timestep,26)
    input_X = keras.layers.Input((timestep,24))
    input_X1 = input_X[:,:,0:22]
    input_X2 = input_X[:,-1,22]
    input_X3 = input_X[:,-1,23]
    X2 = keras.layers.Reshape((1,))(input_X2)
    X3 = keras.layers.Reshape((1,))(input_X3)
    # None, timestep, 20, 14
    X = keras.layers.LSTM(70,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,14))(input_X1)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(70,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,70))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LSTM(70,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,70))(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(70, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(timestep,70))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Concatenate()([X,X2])
    X = keras.layers.Dense(50, activation="relu")(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.Dense(25, activation="relu")(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Concatenate()([X,X3])
    X = keras.layers.Dense(10, activation="softmax")(X)
    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel

def create_ldstm_data_set(df,timestep=5):
    X = []
    Y = []
    df['list'] = df.drop(columns=['VV','STN','V0','rainfall_train.ef_year','day','rainfall_train.ef_hour','class']).apply(lambda x: np.array(x),axis=1)
    sample_weights = 1-df['class'].value_counts(normalize=True).values
    for i in tqdm(range(1,21)):
        stn_df = df[df["STN"] == f"STN0{'%02d' % i}"].copy()
        tmp2 = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['list'].apply(list).values
        max_len = 20
        tmp3 = []
        for i in tmp2:
            n = len(i)
            if max_len == n:
                tmp3.append(np.array(i))
            else:
                #평균값
                tmp3.append(np.vstack([np.array(i),np.array([(np.mean(np.array(i),axis=0))] * (max_len-n))]))
                # DH 빠른 값
                # tmp3.append(np.vstack([np.array(i),np.array([np.array(i[0])] * (max_len-n))]))
                # print(i[0])
                # break
        y_tmp = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['class'].mean().values.astype(int)
        Y.extend(y_tmp)
        m = len(tmp3) - timestep
        tmp4 = []
        for i in range(timestep):
            s = np.array(tmp3[0:i+1])
            tmp4.append(np.vstack([s,np.repeat(tmp3[i].reshape(1,20,16), timestep - (i+1), axis=0)]))
        for i in range(m):
            tmp4.append(np.array(tmp3[i:i+timestep]))
        X.extend(tmp4)
    Y = np.array(Y)
    sample_weights = sample_weights[Y]*10
    return np.array(X),Y,sample_weights

def create_fstm_data_set(df,timestep=5):
    X = []
    Y = []
    df['list'] = df.sort_values(by=['DH']).drop(columns=['VV','STN','V0','rainfall_train.ef_year','day','rainfall_train.ef_hour','class']).apply(lambda x: np.array(x),axis=1)
    sample_weights = 1-df['class'].value_counts(normalize=True).values
    for i in tqdm(range(1,21)):
        stn_df = df[df["STN"] == f"STN0{'%02d' % i}"].copy()
        tmp2 = np.array(list(stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['list'].apply(lambda x: x.iloc[0]).values))
        y_tmp = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['class'].mean().values.astype(int)
        Y.extend(y_tmp)
        tmp3 = tmp2
        # tmp3 = np.hstack([tmp2,talib.SMA(tmp2[:,14],timeperiod=5).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(tmp2[:,14],timeperiod=10).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(tmp2[:,15],timeperiod=5).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(tmp2[:,15],timeperiod=10).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(talib.STDDEV(tmp2[:,14]), timeperiod=5).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(talib.STDDEV(tmp2[:,14]), timeperiod=10).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(talib.STDDEV(tmp2[:,15]), timeperiod=5).reshape(-1,1)])
        # tmp3 = np.hstack([tmp3,talib.SMA(talib.STDDEV(tmp2[:,15]), timeperiod=10).reshape(-1,1)])
        #macd,_,_ =talib.MACD(tmp2[:,14], fastperiod=12, slowperiod=26, signalperiod=9)
        #tmp3 = np.hstack([tmp3,macd.reshape(-1,1)])
        #macd,_,_ =talib.MACD(tmp2[:,15], fastperiod=12, slowperiod=26, signalperiod=9)
        #tmp3 = np.hstack([tmp3,macd.reshape(-1,1)])
        m = len(tmp3) - timestep
        tmp4 = []
        for i in range(timestep):
            s = np.array(tmp3[0:i+1])
            tmp4.append(np.vstack([s,np.repeat(tmp3[i].reshape(1,16), timestep - (i+1), axis=0)]))
        for i in range(m):
            tmp4.append(np.array(tmp3[i:i+timestep]))
        X.extend(tmp4)
    Y = np.array(Y)
    sample_weights = sample_weights[Y]*10
    return np.array(X),Y,sample_weights

def create_dstm_data_set(df,timestep=5):
    X = []
    Y = []
    df['list'] = df.drop(columns=['VV','STN','V0','rainfall_train.ef_year','day','rainfall_train.ef_hour','class']).apply(lambda x: np.array(x),axis=1)
    sample_weights = 1-df['class'].value_counts(normalize=True).values
    for i in tqdm(range(1,21)):
        stn_df = df[df["STN"] == f"STN0{'%02d' % i}"].copy()
        tmp2 = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['list'].apply(list).values
        max_len = 20
        tmp3 = []
        for i in tmp2:
            n = len(i)
            if max_len == n:
                tmp3.append(np.array(i))
            else:
                #평균값
                tmp3.append(np.vstack([np.array(i),np.array([(np.mean(np.array(i),axis=0))] * (max_len-n))]))
                # DH 빠른 값
                # tmp3.append(np.vstack([np.array(i),np.array([np.array(i[0])] * (max_len-n))]))
                # print(i[0])
                # break
        y_tmp = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['class'].mean().values.astype(int)
        Y.extend(y_tmp)
        m = len(tmp3) - timestep
        tmp4 = []
        for i in range(timestep):
            s = np.array(tmp3[0:i+1])
            tmp4.append(np.vstack([s,np.repeat(tmp3[i].reshape(1,20,16), timestep - (i+1), axis=0)]))
        for i in range(m):
            tmp4.append(np.array(tmp3[i:i+timestep]))
        X.extend(tmp4)
    Y = np.array(Y)
    sample_weights = sample_weights[Y]*10
    return np.array(X),Y,sample_weights


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder


def create_all_data_set():
    rain_train = pd.read_csv("./rainfall_train.csv")
    rain_train.columns = [
        'Unnamed: 0',
        'fc_year', 'fc_month', 'fc_day', 'fc_hour',
        'stn4contest', 'dh',
        'ef_year', 'ef_month', 'ef_day', 'ef_hour',
        'v01', 'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09',
        'vv', 'class_interval'
    ]

    # 불필요한 변수 제거
    rain_train.drop(columns=['Unnamed: 0'], inplace=True)

    df = rain_train.copy()
    # -999 값을 NaN으로 변환
    df = df[df['class_interval'] != -999]

    # 월별 누적 일수 계산
    month_to_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(1, 12):
        month_to_day[i] += month_to_day[i-1]
    month_to_day = {idx+2: i for idx, i in enumerate(month_to_day)}
    month_to_day[1] = 0

    # 주기적 특성 추가
    df['day'] = df['ef_month'].apply(lambda x: month_to_day[x]) + df['ef_day']
    df['day_sin'] = np.sin(2*np.pi*df['day']/365)
    df['day_cos'] = np.cos(2*np.pi*df['day']/365)
    # df = df.drop(columns=['ef_month', 'ef_day'])
    
    df['hour_sin'] = np.sin(2 * np.pi * df['ef_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['ef_hour'] / 24)
    # df = df.drop(columns=['ef_hour'])
    # 칼만 필터 적용 (예: v01 변수에 적용)
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.filter(df['v01'].values)
    df['v01'] = state_means.flatten()

    # 이진 분류를 위한 타겟 생성
    df['binary_target'] = df['class_interval'].apply(lambda x: 0 if x == 0 else 1)

    # OneHotEncoder를 사용하여 fc_year 원핫 인코딩
    ohe = OneHotEncoder(handle_unknown='ignore')
    fc_year_encoded = ohe.fit_transform(df[['fc_year']]).toarray()
    fc_year_encoded_df = pd.DataFrame(fc_year_encoded, columns=ohe.get_feature_names_out(['fc_year']))
    df = pd.concat([df, fc_year_encoded_df], axis=1)
    df = df.drop(columns=['fc_year'])

    # 필요한 특성 선택
    features = ['dh', 'v01', 'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'] + list(fc_year_encoded_df.columns)
    target = 'class_interval'
    # df = df[features]
    #중복없이 가장 예보와 예상시간이 가까운 경우 출력
    #데이터 수를 줄이는 알고리즘 따로 정리할 필요가 있음
    # close_tr = df[df['dh']<=12]
    #ef_hour
    df['forecast'] = df['ef_hour'] + df['dh'] 
    df['forecast'] = np.where(df['forecast'] == 24, 0, df['forecast'])
    df['forecast'] = np.where(df['forecast'] > 24, df['forecast'] - 24, df['forecast'])
    # df = df.copy().drop(['ef_hour'], axis=True)
    #언더샘플링을 위한 v00 컬럼생성
    df['v00'] = (df[['v01', 'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09']].sum(axis=1) == 0).astype(int) * 100
    #kalmanfilter
    for var in ['v01', 'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09']:
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        state_means, _ = kf.filter(df[var].values)
        df[f'{var}_kalman'] = state_means.flatten()
    return df

def data_sampling(df):
    #데이터 샘플링 관련
    # 데이터 언더샘플링
    df_class_0 = df[df['class_interval'] == 0]
    df_class_non_0 = df[df['class_interval'] != 0]
    
    # class_interval 값이 0인 데이터의 언더샘플링
    df_class_0_under = df_class_0.sample(len(df_class_non_0), random_state=42)
    
    # 언더샘플링된 데이터와 나머지 데이터를 결합
    df_balanced = pd.concat([df_class_0_under, df_class_non_0])

from util import preprocessing_daegun
if __name__ == "__main__":
    df = preprocessing_daegun
    X,Y, sample_weigts= create_fstm_data_set(df)