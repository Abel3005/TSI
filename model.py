from tqdm import tqdm
import numpy as np
from tensorflow import keras


def dstm_model(timestep=5):
    input_X = keras.layers.Input((timestep,20,14))
    process_channel = []
    for i in range(14):
        # None, timestep, 20
        channel_slice = input_X[:,:,:,i]
        process_timestep = []
        for j in range(timestep):
            #None, 20
            process = keras.layers.Dense(5,activation="sigmoid",input_shape=(20,))(channel_slice[:,j,:])
            process = keras.layers.Dense(3,activation="sigmoid",input_shape=(20,))(process)
            process_timestep.append(process)
        # (5,3)
        process_timestep = keras.layers.Concatenate()(process_timestep)
        process_channel.append(keras.layers.Reshape((timestep,3))(process_timestep))
    #(timestep,)
    X = keras.layers.Concatenate(axis=-1)(process_channel)
    curr_procssing = []
    for i in range(timestep):
        curr = X[:,i,:]
        # (42,)
        curr = keras.layers.Dense(100, activation="tanh")(curr)
        curr = keras.layers.Dropout(0.2)(curr)
        curr = keras.layers.Dense(50, activation="relu")(curr)
        curr = keras.layers.Dropout(0.2)(curr)
        curr = keras.layers.Dense(24, activation="relu")(curr)
        curr = keras.layers.BatchNormalization()(curr)
        curr = keras.layers.Dense(12, activation="relu")(curr)
        curr_procssing.append(curr)
    X = keras.layers.Concatenate()(curr_procssing)
    X = keras.layers.Reshape((timestep,12))(X)
    X = keras.layers.LSTM(15,input_shape=(5,12))(X)
    X = keras.layers.Dense(10, activation="softmax")(X)

    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel

def create_dstm_data_set(df,timestep=5, sample_weight=False):
    X = []
    Y = []
    df['list'] = df.drop(columns=['VV','STN','V0','rainfall_train.ef_year','day','rainfall_train.ef_hour','class']).apply(lambda x: np.array(x),axis=1)
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
                tmp3.append(np.vstack([np.array(i),np.array([(np.mean(np.array(i),axis=0))] * (max_len-n))]))
        y_tmp = stn_df.groupby(by=['rainfall_train.ef_year','day','rainfall_train.ef_hour'])['class'].mean().values.astype(int)
        Y.extend(y_tmp[5:])
        m = len(tmp3) - timestep
        tmp4 = []
        for i in range(m):
            tmp4.append(np.array(tmp3[i:i+timestep]))
        X.extend(tmp4)
    sample_weights = np.ones(len(Y))
    if sample_weight:
        sample_weights[Y !=0] = 3
    return np.array(X),np.array(Y),sample_weights