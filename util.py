import numpy as np

def compute_csi(y_pred,y_true):
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