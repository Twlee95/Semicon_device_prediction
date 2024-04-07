from sklearn.metrics import r2_score

def pred_acc(y_pred, y_true):
    value_ = np.abs((y_true-y_pred)/y_true)
    pred_result = np.where(value_<=0.03, 1, 0)
    return (sum(pred_result)/len(y_pred))*100


def R2_acc(y_true, y_pred):
    return r2_score(y_true, y_pred)

def SL_acc(y_pred, y_true):
    USL = 1.889
    LSL = 1.341
    value_ = np.abs(y_true-y_pred)/(USL-LSL)
    return (sum(value_)/len(y_pred))*100