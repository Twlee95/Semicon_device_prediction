import numpy as np
# 입력 f_pcm_df
def Split_Data(data, seq_len = 10):
    X = data[data.columns[19:]]
    Y = data[['AVG']]
    SP_Data = []
    for idx in range(len(X)-seq_len+1):
        XY = []
        SP_X = X.iloc[idx:idx+seq_len,:].values   # 10x198  IQC Parameter
        PCM_Y = Y.iloc[idx:idx+seq_len,:].values
        
        # SP_ADD = Y.iloc[idx+seq_len+2:idx+seq_len+1+2,:].values
        #print(SP_X)
        #print(SP_ADD)
        #SP_ADD_X = np.append(SP_ADD, SP_ADD[-1]).reshape(seq_len,-1)
        SP_XX = np.concatenate((PCM_Y, SP_X), axis=1)
        SP_Y = Y.iloc[idx+seq_len:idx+seq_len+1,:].values  ## 1x1
        
        XY.append(SP_XX)
        XY.append(SP_Y)
        SP_Data.append(XY)
    return SP_Data

class TS_Dataset:
    def __init__(self, data):
        self.data_ = data
        self.SP_DATA = Split_Data(self.data_)
    
    def __len__(self):
        return len(self.SP_DATA)
    
    def __getitem__(self, idx):
        return self.SP_DATA[idx]