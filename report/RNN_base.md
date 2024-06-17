
## LSTM
### SIMPLE LSTM

```python
model = Sequential()
model.add(LSTM(25, input_shape=(x.shape[1],x.shape[2])))
model.add(Dense(10, activation="softmax"))
```

> 학습 결과

<img src="./images/lstm 학습 그래프.png" />

정확도,CSI 가 낮은 지점에서 수렴
```python
CSI : 0.19572919428309754
```
### DSTM_ DNN_distibuted LSTM Model

**train/test set 분리**

- 시계열 데이터의 특성상 랜덤하게 데이터를 나누는 것이 아닌 시간 순서에 따라 데이터를 구분하는 것이 필요
- B년도의 데이터를 test 데이터를 사용하는 방법 

**시계열로 데이터를 분석**

- 시계열적으로 데이터를 분석하기 위해서 2가지 변수에 따라 집계할 수 있도록 모델 설계
- STN, DH(TM_FC)

>   모델 구조
- DH 처리 : DNN
    - DH 최대 개수 : 20
    - 20* (feature 개수)로 데이터 형식을 맞춤
    - 각 열에 대해서 DNN, LSTM 등으로 feature 생성
    - 1차원으로 flatten()
- STN종류대로 시계열 데이터 셋 생성
    - STN의 종류대로 데이터를 묶음
        - 각 STN에 따라서 시간순으로 정렬된 데이터 묶음이 생성
        - STN의 개수가 20개이면 20개의 데이터 묶음이 생성된다.
    - 위에서 생성된 데이터 묶음을 timestep을 기준으로 묶습니다.

- 샘플 데이터 형식
    - DH에 따라서 구분된 데이터 : (20 * 14)
    - timestep으로 데이터를 묶음 : (5* 20 * 14)

**Default DSTM**
>   모델 구조
```python
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
    X = keras.layers.Dense(30, activation="relu")(X)
    X = keras.layers.Dense(15, activation="relu")(X)
    X = keras.layers.Dense(10, activation="sigmoid")(X)
    X = keras.layers.LSTM(10,input_shape=(5,14*3))(X)
    X = keras.layers.Dense(10, activation="softmax")(X)

    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```
> 학습결과

<img src="./images/LSTM_first.png" />

- 모델이 local optima에 수렴한 것을 볼 수 있음

**DSTM (LSTM 전 Dense Layer 추가)**

```python
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
```
>   학습결과

<img src="./images/second_DSTM.png" />

- 이전보다 깊이 내려가지만 기울기가 급격히 내려가는 구간이 존재한다.
- 검증데이터는 이전과 별반 차이가 없다.

**DSTM : Return_Sequence 추가하여 LSTM layer 추가**
>   모델 구조

```python
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
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(5,12))(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(5,12))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LSTM(50, return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(5,12))(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(50,recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(5,12))(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(10, activation="softmax")(X)

    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```

>   학습 결과
<table><tr><td>
<img src="./images/third_lstm.png" />
</td><td><img src="./images/가중치 다르게 한 third_dstm.png" /></td></tr>
</table>

- 성능이 낮은 위치에서 기울기가 낮아지는 수렴증상을 보임
- 학습을 더 돌려봤을 때, Valid, Train 모두 수렴하는 것으로 나타남

**D(L)STM: LSTM_LSTM 모델**

>   모델 구조

```python
def dstm_model(timestep=5):
    input_X = keras.layers.Input((timestep,20,14))
    # None, timestep, 20, 14
    channel_process = []
    for i in range(timestep):
        channel_ = input_X[:,i,:,:] 
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,14))(channel_)
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        channel_=keras.layers.LSTM(10,return_sequences=True, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        channel_=keras.layers.LSTM(10, recurrent_regularizer=keras.regularizers.l2(0.01),input_shape=(20,10))(channel_)
        #output = None,1,10
        channel_process.append(channel_)
 
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
    X = keras.layers.Dense(10, activation="softmax")(X)
    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```
>   학습결과
<table>
<tr>
<td><img src="./images/D(L)STM_First.png" /></td>
<td><img src="./images/D(L)STM second.png" /><td>
</tr>
</table>

- validation의 변동이 진동됨을 볼 수 있음.
- loss가 0.5 수준에서 과적합이 진행된다고 생각

### RBPLSTM: 무강수/강수 분류모델을 통해서 데이터 추가하는 방식
- 회귀모델: Randomforest(n_estimator=60 ,random_state=42) 
    - 독립변수: V1-9, DH, hour(sin),hour(cos),day(cos),day(sin)
    - 종속변수: VV
- 분류모델: Randomforest(n_estimator=60, random_state=42)
    - 독립변수: V1-9, DH, hour(sin),hour(cos),day(cos),day(sin)
    - 종속변수 : class
- class별로 데이터 분포에 따른 weight 줌

<table>
<tr><td style="font-weight:bold">머신러닝 알고리즘 성능</td></tr>
<tr>
<td>

    실 강수 데이터
    mean_absolute_error: 1.629823192725517
    mean_squared_error : 25.65344261132864

</td>
<td>

                precision    recall  f1-score   support
       False       0.63      0.33      0.43     90625
        True       0.86      0.96      0.91    395311
    accuracy                           0.84    485936
    macro avg       0.75      0.64      0.67    485936
    weighted avg       0.82      0.84      0.82    485936

</td>
</tr>
</table>

- 무강수/강수(분류모델을 통해서 나온 결과) 확률 추가
- Dense Latyer 층에서 실강수량(분류모델을 통해서 나온 결과) 평균 추가

>   학습 결과

<table>
<tr><td><img src="./images/RBPSTM_csi.png" /></td>
<td><img src="./images/RBPSTM_accuracy.png" /></td>
<td><img src="./images/RBPSTM_loss.png" /></td></tr>
</table>

**Dense 층에 머신러닝 결과를 적용**

>   모델 구조
```python
def dstm_model(timestep=5):
    input_X = keras.layers.Input((timestep,20,16))
    input_X1 = input_X[:,:,:,0:14]
    input_X2 = input_X[:,:,:,14]
    input_X3 = input_X[:,:,:,15]
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
    X1 = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(input_X2)
    X1 =keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(X1)
    X2 =keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(input_X3)
    X2 =keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(X2)
 
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
    X1 =keras.layers.Reshape((1,))(X1)
    print(X1.shape)
    X2 =keras.layers.Reshape((1,))(X2)
    X = keras.layers.Concatenate(axis=-1)([X,X1,X2])
    X = keras.layers.Dense(10, activation="softmax")(X)

    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```
> 학습결과

<table>
<tr><td><img src="./images/Dense_ML_first.png" /></td><td><img src="./images/Dense_ML_second_accuracy.png" /></td></tr>
<tr><td><img src="./images/Dense_ML_first_csi.png" /></td><td><img src="./images/Dense_ML_second_csi.png" /></td></tr>
<tr><td><img src="./images/Dense_ML_first_loss.png" /></td><td><img src="./images/Dense_ML_first_loss.png" /></td></tr>
</table>

- Dense 층에 머신러닝 결과를 적용하니, Validation과  Trainning CSI 점수가 안정적으로 0.1까지 올라감
- 0.1 이상까지 학습이 진행되지는 않음

**RBPLSTM: Regression 모델 제외**

```python
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
```
>   학습결과

<img src="./images/D(L)STM_final.png" />


### FSTM_ Fast Data LSTM Model
- 시간당 하나의 데이터를 선별
- 해당 시간 중 가장 dh가 작은 데이터를 사용

<img src="./images/DH 영향도 확인.png" />

DH가 작을수록 데이터의 신뢰성이 높아진다는 것을 가설로 함.

```python
def dstm_model(timestep=5):
    input_X = keras.layers.Input((timestep,20,14))
    process_channel = []
    for i in range(14):
        # None, timestep, 20
        channel_slice = input_X[:,:,:,i]
        process_timestep = []
        for j in range(timestep):
            #None, 20
            process = keras.layers.Dense(10,activation="sigmoid",input_shape=(20,))(channel_slice[:,j,:])
            process = keras.layers.Dense(5,activation="relu",input_shape=(50,))(process)
            process = keras.layers.Dense(3,activation="sigmoid",input_shape=(20,))(process)
            process_timestep.append(process)
        # (5,3)
        process_timestep = keras.layers.Concatenate()(process_timestep)
        process_channel.append(keras.layers.Reshape((timestep,3))(process_timestep))
    #(timestep,)
    X = keras.layers.Concatenate(axis=-1)(process_channel)
    X = keras.layers.Dense(30, activation="relu")(X)
    X = keras.layers.Dense(15, activation="relu")(X)
    X = keras.layers.LSTM(15,input_shape=(5,14*3))(X)
    X = keras.layers.Dense(10, activation="softmax")(X)

    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```
> 학습 결과

<table>
<tr><td><img src="./images/fstm_first_accuracy.png" /></td>
<td><img src="./images/fstm_first_csi.png" /></td>
<td><img src="./images/fstm_first_loss.png" /></td></tr>
</table>

**머신러닝 결과 후처리**

>   모델 구조

```python
def fstm_model(timestep=5):
    #(None,Timestep,26)
    input_X = keras.layers.Input((timestep,16))
    input_X1 = input_X[:,:,0:14]
    input_X2 = input_X[:,-1,14]
    input_X3 = input_X[:,-1,15]
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
    X = keras.layers.Concatenate()([X,X2,X3])
    X = keras.layers.Dense(10, activation="softmax")(X)
    lmodel = keras.Model(inputs=input_X, outputs=X)
    return lmodel
```

>   학습결과

<img src="./images/fstm 결과.png" />

- 머신러닝 결과를 적용해도 학습이 잘 되지는 않음


**FSTM: 기술지표 데이터 추가**

>   모델 구조

```python
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
```

>   학습 결과
<img src="./images/FSTM_기술지표 추가 결과.png" />

- 기술 지표 추가시 데이터가 0으로 맞춰지는 현상 발생
- 따로 기술 지표를 추가한지 않는 방향

### TSTM Teaching LSTM 

- 학습이 제대로 이루어지지 않고 있다고 판단
- Encoder-Decoder의 Teaching 개념을 사용하여 모델을 설계
- 뒷부분의 LSTM의 각 단계에서 입력 값으로 정답 데이터를 주고 학습
- 어느 정도 학습이 되었다고 판단되었을 때, encoder의 학습을 멈추고 Decoder layer을 encoder output으로 학습
- 회귀모델(종속변수: 실강수량)으로 설계
    - scaling을 StandardScaler를 사용할 때, 모든 값이 0으로 되는 현상발생
    - minmax sclaer를 사용하기에는 각 연도별, min-max 값의 편차가 심함
    - 스케일링 없이 진행

**encode_decode 모델**

```python
from tensorflow import keras
timestep=5
input_X = keras.layers.Input((timestep,20,15))
input_Y = keras.layers.Input((timestep,1))
#각 DH 마다의 가중치에 따라 feature 생성
unit = 1
channel_process = []

# 인코딩 과정
for i in range(timestep):
    _channel = keras.layers.Lambda(lambda x : x[:,i,:,:])(input_X)
    _channel = keras.layers.LSTM(100, input_shape=(20,15))(_channel)
    channel_process.append(_channel)
encode_X = keras.layers.Concatenate()(channel_process)
encode_X = keras.layers.Reshape((timestep,100))(encode_X)
encode_X = keras.layers.LSTM(100, return_sequences=True, input_shape=(timestep,))(encode_X)
encode_X = keras.layers.LSTM(50, return_sequences=True, input_shape=(timestep,))(encode_X)
encode_X = keras.layers.LSTM(10, return_sequences=True, input_shape=(timestep,))(encode_X)
encode_out,h,c = keras.layers.LSTM(1, return_sequences=True, return_state=True, input_shape=(timestep,))(encode_X)
# 디코딩 과정
d_h = keras.layers.Reshape((1,))(h[:,-1])
d_c = keras.layers.Reshape((1,))(c[:,-1])
d_o = keras.layers.Reshape((1,1))(encode_out[:,0])
encode_state = [d_h,d_c]
# 비교 (timestep, 10)
DX = keras.layers.Concatenate(axis=1)([d_o,input_Y[:,:-1]])
decoder_lstm = keras.layers.LSTM(unit, return_sequences=True)
DX = decoder_lstm(DX,initial_state=encode_state)
DX = keras.layers.Flatten()(DX)
```
>   학습 결과
<img src="./images/tstm_result.png" />
- 회귀 모델로는 전혀 학습되지 않음을 볼 수 있다.

- decoder의 복잡하지 않은 모델가 회귀 데이터의 다양한 범위를 반영하지 못하는것으로 판단