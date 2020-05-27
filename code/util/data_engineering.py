import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import PredefinedSplit


def train_test_validation_split(df, test_size=0.25, validation_size=0, target="future_price_movement_121d", exclude_columns=["open", "close", "high", "low", "volume", "middle"]):
    x_columns = [ col for col in df.columns if col not in exclude_columns + [target] ]
    # calc train- and test size
    train_set_size = int( df.shape[0] * (1 - test_size) )
    test_set_size = df.shape[0] - train_set_size
    # take x-matrix
    x_train = df[ x_columns  ][ :train_set_size ]
    x_test = df[ x_columns ][ train_set_size: ]                         
    # take target variable
    y_train = df[ target ][ :train_set_size ]
    y_test = df[ target ][ train_set_size: ]
    if validation_size:
        # make split-object
        validation_set_size =  int(x_train.shape[0] * validation_size)
        non_validation_set_size =  x_train.shape[0] - validation_set_size
        validation_split = PredefinedSplit(test_fold=[ -1 ] * non_validation_set_size + [ 1 ] * validation_set_size)
        return x_train, x_test, y_train, y_test, validation_split
    else:
        return x_train, x_test, y_train, y_test


def up_down_indicator(x):
    if pd.isnull(x):
        return x
    elif x > 0:
        return 1
    elif x <= 0:
        return 0


def top_long_short_return(row, threshold=0.5):
    up_max_pair = row["up_prob"].argmax()
    down_max_pair = row["down_prob"].argmax()
    if threshold:
        # obtain max prob for up-movement and its pair
        up_max_pair = row["up_prob"].argmax()
        up_max_prob = row["up_prob"].max()
        # set return to 1 if up-prob below threshold else to its respective long return
        long_return = 0 if up_max_prob < threshold else row["future_middle_return_121d"][ up_max_pair ]
        # obtain max down for up movement and its pair
        down_max_pair = row["down_prob"].argmax()
        down_max_prob = row["down_prob"].max()            
        # set return to 1 if prob below threshold else to its respective short return
        short_return = 0 if down_max_prob < threshold else -row["future_middle_return_121d"][ down_max_pair ]
    else:
        # set long- and -return to its value for the respective best pair  
        short_return = row["future_middle_return_121d"][ up_max_pair ]
        long_return = -row["future_middle_return_121d"][ down_max_pair ]
                          
    return 1 + short_return, 1 + long_return


