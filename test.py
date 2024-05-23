import tensorflow as tf
import numpy as np
import pandas as pd
import os
from metric import CSImetric

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

os.chdir('/home/encore/workspace/TSI')

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='true_positives', initializer='zeros')
        self.fp = self.add_weight(name='false_positives', initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', initializer='zeros')
        self.tn = self.add_weight(name='true_negatives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.tn.assign_add(tn)
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score
    
    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        self.tn.assign(0.0)




def cretae_data():
    rainfall_train = pd.read_csv('./rainfall_train.csv')
    rainfall_train.drop(columns=['Unnamed: 0'],inplace= True)
    df = pd.concat([pd.read_csv('./daegun_first.csv'),rainfall_train[['rainfall_train.dh','rainfall_train.ef_month','rainfall_train.ef_day','rainfall_train.ef_hour']]],axis=1)
    df = df.drop(columns=['TM_FC','TM_EF','EF_class','STN'])
    null_df = df[df['class'] == -999]
    df = df[df['class'] != -999]
    month_to_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(1,12):
        month_to_day[i] += month_to_day[i-1]
    month_to_day = {idx+2 : i for idx, i in enumerate(month_to_day)}
    month_to_day[1] = 0
    df['day_sin'] = np.sin(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df['day_cos'] = np.cos(2*np.pi*df['rainfall_train.ef_month'].apply(lambda x: month_to_day[x]) + df['rainfall_train.ef_day']/365)
    df =df.drop(columns=['rainfall_train.ef_month','rainfall_train.ef_day'])
    df['hour_sin'] = np.sin(2 *np.pi * df['rainfall_train.ef_hour'] /24)
    df['hour_cos'] = np.cos(2 *np.pi * df['rainfall_train.ef_hour'] /24)
    df = df.drop(columns=['rainfall_train.ef_hour'])
    # 전처리
    yr= df.pop('VV')
    yc =df.pop('class')
    return train_test_split(df,yc,test_size=0.2,shuffle=True,random_state=42) 

def dnn_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(30, activation='sigmoid',input_shape=(16,)))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(100, activation='relu',input_shape=(35,)))
    model.add(keras.layers.Dense(1000, activation='relu',input_shape=(35,)))
    model.add(keras.layers.Dense(20, activation='relu',input_shape=(34,)))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model




#scaler
#V1-9 : 1/100
#DH: Min-Max sclaer
if __name__ == "__main__":
    x,x_v, y,y_v = cretae_data()
    scaler = MinMaxScaler()
    scaler.fit(x[['DH']])
    x[['DH']]=scaler.transform(x[['DH']])
    x_v[['DH']] = scaler.transform(x_v[['DH']])
    x_v[[f"V{i}" for i in range(1,10)]] = x_v[[f"V{i}" for i in range(1,10)]] / 100
    x_v[[f"V{i}" for i in range(1,10)]] = x_v[[f"V{i}" for i in range(1,10)]] / 100
    model = dnn_model()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy',CSImetric()])
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    history = model.fit(x, y, epochs=20, verbose=1, validation_data=(x_v, y_v),callbacks=[checkpoint_cb,early_stopping_cb])




