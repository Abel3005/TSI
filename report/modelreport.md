# 모델

## 무강수/강수 분류 모델

### RandomForest 무강수/강수 분류**
>   독립변수
- DH
- V1-9
- hour: sin, cos
- day: sin, cos

>   종속변수
- class : 강수계급
- VV : 실 강수량

>   분류 결과

```python
                precision   recall    f1-score support

       False       0.69      0.45      0.55     45133
        True       0.90      0.96      0.93    244620

    accuracy                           0.88    289753
   macro avg       0.80      0.71      0.74    289753
weighted avg       0.87      0.88      0.87    289753
```


### DNN (가중치 설정)을 통한 분류**
```python
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation='sigmoid',input_shape=(14,)))
model.add(keras.layers.Dense(100, activation='relu',input_shape=(35,)))
model.add(keras.layers.Dense(20, activation='relu',input_shape=(34,)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```
>   하이퍼 파라미터

- optimizer : adam
- loss_func : Cross_Entropy

>   결과

<img src="../images/DNN 강수_무강수.png" />

- 학습 시간이 오래 걸릴 뿐만이 아니라 오래걸림.
- DNN 보다는 머신러닝과 V0 임계값을 이용하여 데이터 분류하는 방법 사용.


> 머신러닝 분류 결과
```python
랜덤포레스트 적용
전체 데이터 개수: 1448762
분류된 무강수 데이터 개수: 1241310
분류된 무강수 중 실제 무강수 데이터 개수: 1213706
----------------------------------------
남은데이터중 데이터 무강수 개수: 9609
남은데이터중 데이터 강수 개수: 197843
남은데이터중 데이터 비율: 0.048568814666174694
```

**분류된 데이터 분포**

<table>
<tr><td>무강수 데이터</td><td></td><td>강수 데이터</td><td></td></tr>
<tr><td>class</td><td>개수</td><td>class</td><td>개수</td></tr>
<tr><td>0</td>    <td>1213706</td><td>0</td>    <td>9609</td></tr>
<tr><td>1</td>    <td>   3790</td><td>1</td>    <td>17772</td></tr>
<tr><td>2</td>    <td>   5375</td><td>2</td>    <td>28679</td></tr>
<tr><td>3</td>    <td>   4424</td><td>3</td>    <td>26268</td></tr>
<tr><td>4</td>    <td>   4229</td><td>4</td>    <td>29198</td></tr>
<tr><td>5</td>    <td>   4537</td><td>5</td>    <td>36870</td></tr>
<tr><td>6</td>    <td>   2528</td><td>6</td>    <td>25436</td></tr>
<tr><td>7</td>    <td>   1671</td><td>7</td>    <td>19497</td></tr>
<tr><td>8</td>    <td>    610</td><td>8</td>    <td>7577</td></tr>
<tr><td>9</td>    <td>    440</td><td>9</td>    <td>6546</td></tr>
</table>

## 강수 클래스 구분 모델 학습

### Random Forest

- 최적의 n_estimator 60으로 결정

> A년도 dh = 0인 데이터에 대한 모델 평가
<img src="../images/변수 영향도 결정을 위한 모델 성능 수치 가시화.png" />

> dh = 0인 강수 데이터에 대해서 성능 평가
<img src="../images/(모든년도)변수 영향도 결정을 위한 모델 성능 수치 가시화.png" />

>   강수데이터에 대해서 Random Forest 성능 평가 
<img src="../images/최적의 RandomForeset.png" />

### cluster-bassd machine learning model
- 데이터의 분포를 나눠서 머신러닝으로 분류하도록 하는 알고리즘

>   프로세스
1. 비슷한 데이터 분포를 띄는 것끼리 데이터를 나눔
2. 나눠진 각 클러스터 마다 앙상블 모델(랜덤 포레스트 적용)

### 클러스터 나누기
- 별개의 시간과 장소에 대한 데이터만 남기도록 데이터를 전처리
    - dh가 가장 작은 것으로 판단