def compute_csi(y_pred,y_true):
    # compute csi
    h =((y_true != 0) & (y_pred == y_true)).sum()
    f = (y_pred != y_true).sum()
    return h/(h+f)