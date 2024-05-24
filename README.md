# TSI (2024년 날씨 빅데이터 컨테스트 공모)

## 수치모델 앙상블을 활용한 강수량 예측

3시간 강수량 확률(앙상블 모델들을 통해서 나온 값)을 사용하여 누적 강수량을 예측

정확한 강수량을 예측하는 회귀 분석 문제가 아닌 강수량의 정도에 따른 구간별 계급을 예측하는 분류분석 문제에 해당

### 데이터 수집

날씨마루에서 제공
- Rstudio를 통해서 HIVE에서 데이터 다운로드
    - dbGetQuery(conn, "select * from rainfall_train)
    - scv.write()
- CSV 파일로 구글 드라이브로 옮김




### 데이터 분석

 >   변수 파악

**분류 계급 10개** - 강수량이 0.1mm 미만인 경우는 따로 변수 추가 안함

<table>
<tr><td>v01</td><td>0.1mm 이상 강수</td><td>V02</td><td>0.2mm 이상 강수</td></tr>
<tr><td>v03</td><td>0.5mm 이상 강수</td><td>V04</td><td>1.0mm 이상 강수</td></tr>
<tr><td>v05</td><td>2.0mm 이상 강수</td><td>V06</td><td>5.0mm 이상 강수</td></tr>
<tr><td>v07</td><td>10.0mm 이상 강수</td><td>V08</td><td>20.0mm 이상 강수</td></tr>
<tr><td>v09</td><td>30.0mm 이상 강수</td><td></td><td></td></tr>
</table>

---

>   물리적 변수 범주화

<table>
<tr><td>TM_FC</td><td>DH, TM_EF</td></tr>
<tr>
    <td>

#### rainfall_train.fc_year
- 3년 간의 강수량임을 확인 
- B년도의 분포가 적은편

#### rainfall_train.fc_month
- 5-9월 간의 데이터

#### rainfall_train.fc_hour
- 오전 9시, 오후 9시 발표된 데이터

#### rainfall_train.stn4contest
- 총 20개 지점
- 각 지점마다 약 72000개의 데이터
    </td>
    <td>

    - DH 3의 배수를 가지며 최대 240읠 값을 가짐
    
    - <span style="align:center;font-weight:bold;">TM_EF = TM_FC + DH</span> 
    
    - 발표 시각으로 부터 몇시간 떨어져 있는지를 DH 정보가 나타내고 TM_EF는 그것을 반영한 시간입니다.
    - 발표시간이 12시간 간격임을 고려할 때 TM_EF의 값은 여러개가 됨을 알 수 있습니다.
    - 이전 발표시각에서 예측한 시간과 다음 발표시간에서 예측한 시간이 겹쳤기 때문입니다.
    - 그 예측된 시간에 따라서, 실강수량이 제시됩니다.
    ```python
    (df.groupby(by=['TM_EF','STN'])['VV'].nunique() > 2).sum()
    >> 0
    ```
    </td>
</tr>
<tr><td>(V01 - V09)</td><td>강수계급(classs_interval)</td></tr>
<tr>
<td>

- 무 강수를 제외한 누적확률 값으로 int형으로 구성
- 백분율을 기준으로 표현 되며, 각 단계별 확률을 구하기 위해서는 각 값의 차이 값을 골라야 한다.
</td>
<td>

- 10개의 클래스를 가지고 있습니다.
- 각각의 클래스는 불균형적으로 이루어져 있습니다.
- 결측치가 존재
</td>
</tr>

</table>

### 데이터가 전체적으로 불균형(무강수/강수)

<img src="./images/학습데이터 분포 확인.png" />

>   무강수 데이터 분포 확인

<table>
<tr><td><img src="./images/지점별 무강수 데이터 분포.png" /></td><td><img src="./images/월별 무강수 데이터 분포.png" /></td></tr>
<tr><td><img src="./images/시간별 무강수 데이터 분포확인.png" /></td><td><img src="./images/일별 무강수 데이터 분포.png"</td></tr>
</table>

>  각 클래스 군별로 데이터 분포 확인

<img src="./images/지점별 강수데이터 분포.png" />
<img src="./images/월별 강수 데이터 분포.png" />
<img src="./images/시간별 강수 데이터분포.png" />
<img src="./images/일별 강수 데이터 분포.png" />



## 모델 

### 모델 검증

>   평가 방법

- 무강수 데이터의 개수: 약 84% 강수 데이터 개수 : 약 16%
- 'V0'의 크기가 큰  불균형 데이터이므로, class 별 F1 score 확인
-  $CSI = \frac{H}{H+F+M}$

### 데이터 전처리

#### max([V0-V9])를 이용한 데이터 분류

>   확률이 가장 높은 값을 클래스로 정하였을 때(:예측 모델: max([V0-V9]))

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
target_name = [f"V{i}"for i in range(10)]
print(classification_report(df['EF_class'],df['class'],target_names=target_name))
```
```python
              precision    recall  f1-score   support

          V0       0.94      0.88      0.91   1307439
          V1       0.00      0.05      0.01      2262
          V2       0.06      0.05      0.05     35990
          V3       0.04      0.07      0.05     16781
          V4       0.04      0.10      0.06     13705
          V5       0.17      0.15      0.16     47267
          V6       0.09      0.16      0.11     15991
          V7       0.08      0.19      0.11      8668
          V8       0.01      0.23      0.01       251
          V9       0.02      0.30      0.03       408

    accuracy                           0.81   1448762
   macro avg       0.14      0.22      0.15   1448762
weighted avg       0.86      0.81      0.83   1448762
```
>   DH 영향도 확인

<img src='./images/DH 영향도 확인.png' />

```python
df.groupby(by=['TM_EF','VV'])['DH'].nunique().describe()
```
```python
count    14737.000000
mean        18.800163
std          3.582445
min          1.000000
25%         20.000000
50%         20.000000
75%         20.000000
max         20.000000
Name: DH, dtype: float64
```

> 데이터 전처리

- 월과 일의 데이터 컬럼을 합쳐서 365일을 기준으로 주기성을 넣어주기 위하여 sin 함수와 cos 함수를 사용하여 전처리
- 시간 역시 동일하게 24시간의 주기성을 넣어줌
- DH 는 MinMaxscaler를 사용하여 scale 해줌
- 각각의 확률값은 백분율에서 0~1 사이의 값으로 바꿔줌
- train 데이터와 test 데이터의 지점 데이터가 달라서 STN을 쓰지 않는다.


> 불균형 데이터 처리
<table>
<tr>
<td><img src="./images/무강수 추정확률 분포에 따른 무강수 데이터 분포.png" /></td>
<td><img src="./images/V0임계값에 따른  Precision 가시화.png" /></td>
</tr>
</table>

> 불균형 데이터 처리 정리

**1. V0의 임계값을 통해 무강수/강수 분류**

<img src="./images/V0임계값에 따른  Precision 가시화.png" />

**2. 랜덤포레스트 기법을 통하여 무강수/강수 분류**
```python
                precision   recall    f1-score support

       False       0.69      0.45      0.55     45133
        True       0.90      0.96      0.93    244620

    accuracy                           0.88    289753
   macro avg       0.80      0.71      0.74    289753
weighted avg       0.87      0.88      0.87    289753
```

**3.DNN (가중치 설정)을 통한 분류**
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

<img src="./images/DNN 강수_무강수.png" />

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

## 모델 학습

### LSTM

> 강수로 판별된 데이터 학습 모델

```python
model = Sequential()
model.add(LSTM(25, input_shape=(x.shape[1],x.shape[2])))
model.add(Dense(10, activation="softmax"))
```

> 처리 결과

```python
CSI: 0.997524
```
> 전체 데이터에 대해서 점수
