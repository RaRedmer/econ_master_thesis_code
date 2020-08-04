#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import PredefinedSplit
from scipy.stats.mstats import gmean
from copy import deepcopy
from dataclasses import dataclass
import glob

#%%      
class Backtest:
    def __init__(self, test_df, prediction_prob, target_column, return_column, price_column, include_volume=False):
        self.test_df = test_df
        self.prediction_prob = prediction_prob
        self.return_column = return_column
        self.price_column = price_column
        self.make_trading_signals(include_volume=include_volume)
        self.trading_decisions = None
        self.returns = []

    def make_trading_signals(self, include_volume=False):
        """ Transpose data such that for each row, containing a minute-bin, 
            one has future return min if traded and up- and down-probability for each pair.
        """
        print("Make Trading-Signals")
        self.trading_signals = (self.test_df[ [self.return_column, self.price_column] ] 
                                if include_volume 
                                else  self.test_df[ [self.return_column, "volume_scaled", self.price_column] ])
        self.prob_columns = ["down_prob", "up_prob"] if self.prediction_prob.shape[1] == 2 else ["down_prob", "stable_prob", "up_prob"]

        for index, column in enumerate(self.prob_columns):
            self.trading_signals[column] = [ x[index] for x in self.prediction_prob ]
        
        self.trading_signals = ( 
            self.trading_signals
            .groupby( ["time", "pair"] )
            .first()
            .unstack() 
        )
        for column in ["down_prob", "up_prob"]:
            self.trading_signals[ f"max_{column}" ] = self.trading_signals[ f"{column}" ].max(axis=1)
            self.trading_signals[ f"max_{column}_pair" ] = self.trading_signals[ f"{column}" ].idxmax(axis=1)

        # self.trading_signals = self.trading_signals.head(400) #### REMOVE

    def get_equal_weight_holding_returns(self, transaction_cost=0.003):
        pairs = self.test_df.index.get_level_values("pair").unique().values
        data = {}
        for pair in pairs:
            pair_df = self.test_df[ self.test_df.index.get_level_values("pair") == pair ].sort_index()
            start_price = pair_df.iloc[0]["middle_median"]
            end_price = pair_df.iloc[-1]["middle_median"]

            short_return = -(end_price - start_price) / start_price
            long_return = (end_price - start_price) / start_price
            data[ pair ] = [ short_return, long_return]
        return pd.DataFrame.from_dict(data, orient="index", columns=[ "long_return", "short_return"]) - transaction_cost
                   
    def conduct_top_short_long_alg(self, threshold=0, positions=120, transaction_cost=0.003, delta=120, position_type="both", skip_zero_volume=True):
        """ Simulate trading on the submitted trading data and generated trading-signals by the respective model.
            DISCLAIMER: As of right now, only single long position.
            (outlined in more detail in chapter ???)
        
            Args:
                positions (int): Number of maximum concurrent active positions of long or short 
                                (if positions=120 then there are at maximum 240 active long and 
                                short position in total)

                

            Raises:
                ValueError: trading_decisions is not a pandas-DataFrame 
                            i.e. no trades have been made in the backtest

            Returns:
                pd.DataFrame: Returns for each trade made for each position
        """        
        ts = TopLongShortStrategy(trading_signals=self.trading_signals, positions=positions, delta=delta)
        position_return_df =  ts.conduct_strategy(threshold, skip_zero_volume, transaction_cost, position_type=position_type)
        return position_return_df


class TopLongShortStrategy:
    def __init__(self, trading_signals, positions=120, delta=120):
        self.trading_signals = trading_signals
        self._max_row_num = self.trading_signals.shape[0]
        self.states = len(set(x[0] for x in self.trading_signals.columns if x[0] in ["down_prob", "stable_prob", "up_prob"]))
        self.positions = positions
        self.delta = delta
        self._start_row_num = 0
        self.decision_function = self._top_2state_decision if self.states == 2 else self._top_3state_decision
        self._return_position = {}

    def _top_2state_decision(self, pair, threshold=1/2):
        prob_col = "up_prob" if self._position_state["type"] == "long" else "down_prob"
        max_prob = self._bin[ prob_col ][ pair ][0]
        if max_prob > threshold:
            return True
        else:
            return False   

    def _top_3state_decision(self, pair, threshold=1/3):
        movement = "up" if self._position_state["type"] == "long" else "down"
        pair_stable_prob = self._bin["stable_prob"][ pair ][0] 
        other_movement = "up" if movement == "down" else "down"
        other_movement_prob = self._bin[f"{other_movement}_prob"][ pair ][0] 
        if self._bin[f"max_{movement}_prob"][0] > np.nanmax( [ threshold, pair_stable_prob, other_movement_prob ] ):
            return True
        else:
            return False

    def _get_return(self):
        """ Calculate return of current position depending on the type of position ie long or short
        """
        if self._position_state["type"] == "long":
            return (
                (self._position_state["close_price"] - self._position_state["open_price"])
                / self._position_state["open_price"] 
            )
        elif self._position_state["type"] == "short":
            return (
                (self._position_state["open_price"] - self._position_state["close_price"])
                / self._position_state["close_price"] 
            )

    def _close_position(self):
        """ Calculate return of current position and then close it ie reset the position's attributes  """
        # skip min-bins until volume is not zero
        self._skip_to_non_zero_volume_row(self._position_state["pair"])

        self._position_state["close_price"] = self._bin["middle_median"][ self._position_state["pair"] ][0]
        self._position_state["close_return"] = self._get_return()
        self._position_state["close_time"] = self._bin.index[0]
        self._position_state["close_row_num"] = self._row_num
        self._position_state["duration"] = self._position_state["close_row_num"] - self._position_state["open_row_num"]
        # add closed position info to list
        self._total_position_data.append( deepcopy(self._position_state) )
        # reset position
        self._position_state.update( 
            { key: "" for key, _ in self._position_state.items() if key not in ["position_num", "type"] } 
        )

    def _skip_to_non_zero_volume_row(self, max_prob_pair):
        if self._skip_zero_volume:
            volume = self._bin["volume_scaled"][ max_prob_pair ][0]
            while self._row_num < self._max_row_num and not volume:
                # get current minute-bin data
                self._bin = self.trading_signals.iloc[ [ self._row_num ] ]
                # get current traded volume
                volume = self._bin["volume_scaled"][ max_prob_pair ][0]
                if not volume:
                    self._row_num += 1

    def _open_position(self, new_pair):
        # # skip min-bins until volume is not zero
        # self._skip_to_non_zero_volume_row(new_pair)
        # update pair
        self._position_state["pair"] = new_pair
        # update price
        self._position_state["open_price"] = self._bin["middle_median"][ self._position_state["pair"] ][0]
        # update row number
        self._position_state["open_row_num"] = self._row_num
        # update position opening
        self._position_state["open_time"] = self._bin.index[0]

    def _open_position_under_constraint(self, new_pair):
        # go to next min-bin to enter new position (execution gap)
        self._row_num += self._execution_gap
        self._bin = self.trading_signals.iloc[ [ self._row_num ] ]
        if self._skip_zero_volume:
            volume_max_pair = self._bin["volume_scaled"][ new_pair ][0]
            # check if volume is available for signaled pair
            if volume_max_pair:
                # open position if volume was traded for the respective pair
                self._open_position(new_pair=new_pair)
        else:
            self._open_position(new_pair=new_pair)

    def conduct_strategy(self, threshold=0, skip_zero_volume=False, transaction_cost=0.003, execution_gap=1, position_type="both"):
        threshold = 1 / self.states if not threshold else threshold
       
        self._total_position_data = []
        self._transaction_cost = transaction_cost
        self._skip_zero_volume = skip_zero_volume
        self._execution_gap = execution_gap
        types = {
            "long": ["long"],
            "short": ["short"],
            "both": ["short", "long"],
        }
        for _type in types[ position_type ]:
            self._start_row_num = 0
            movement = "up" if _type == "long" else "down"
            for position_num in range(1, self.positions + 1):
                print("Position:", position_num, "Type:", _type)
                # self._start_row_num = 0 if not self._start_row_num else self._start_row_num
                self._position_state = dict(
                    position_num=position_num,
                    type=_type,
                    pair = "",
                    open_row_num="",
                    open_time = "",
                    open_price = "",
                    duration = "",
                    close_row_num="",
                    close_time="",         
                    close_price = "",
                    close_return = "",
                )
                self._prob_signal = False
                # determine start_row_num, last_price and last_pair
                self._row_num = self._start_row_num + 1
                if self._skip_zero_volume:
                    while self._row_num < self._max_row_num and not self._position_state["pair"]:
                        # update bin
                        self._bin = self.trading_signals.iloc[ [ self._row_num ] ]
                        # get pair for which hast the max up-prob
                        max_prob_pair = self._bin[f"max_{movement}_prob_pair"][0]
                        # determine whether the pair fulfills prob-conditions
                        self._prob_signal = self.decision_function(pair=max_prob_pair, threshold=threshold)
                        # open new position next bin, if any volume is traded
                        self._open_position_under_constraint(max_prob_pair)
                else:
                    # update bin
                    self._bin = self.trading_signals.iloc[ [ self._row_num ] ]
                    # get pair for which hast the max up-prob
                    max_prob_pair = self._bin[f"max_{movement}_prob_pair"][0]
                    # open position
                    self._open_position(max_prob_pair)
                # save start row
                self._start_row_num = self._row_num          

                print(f'Postion: {position_num}, Start-datetime: {self._position_state["open_time"]} (row number: {self._start_row_num})')
                # skip next delta min-bins
                self._row_num += self.delta
                while self._row_num < self._max_row_num - self.delta:
                    # get current volumes and prob
                    self._bin = self.trading_signals.iloc[ [ self._row_num ] ]
                    # get pair which has the max up-prob
                    max_prob_pair = self._bin[f"max_{movement}_prob_pair"][0]            
                    # determine whether the pair fulfills prob-conditions
                    self._prob_signal = self.decision_function(pair=max_prob_pair, threshold=threshold)
                    # check if prob-condition is fulfilled 
                    if self._prob_signal:
                        # if not active position, but good prob-signal, then open new position
                        if not self._position_state["pair"]:
                            # open new position next bin, if any volume is traded
                            self._open_position_under_constraint(max_prob_pair)

                        # if active position's pair is not the same as the suggested position by prob-signal, then close position
                        elif max_prob_pair != self._position_state["pair"]:
                            # close position
                            self._close_position()
                                    
                    else:
                        # if active position, but bad prob-signal, then close position
                        if self._position_state["pair"]:
                            # if not skip_zero_volume
                            # close position
                            self._close_position()
                            
                    if self._position_state["pair"]:
                        # skip next delta minute-bins in all cases to contemplate new trading decision, thus remaining in active position
                        self._row_num += self.delta
                    else:
                        self._row_num += 1
                # if a position is still active, close it
                if self._position_state["pair"]:
                    self._close_position()

        trading_results_df = pd.DataFrame(self._total_position_data).sort_values( ["position_num", "type"] )
        trading_results_df["transaction_cost"] = self._transaction_cost
        return trading_results_df
    

@dataclass    
class Position:
    position_num: int
    type: str
    pair: str = np.NaN
    transaction_cost: int = 0.003
    open_row_num: int = np.NaN
    open_time: datetime = np.NaN
    open_price: float = np.NaN
    duration: int = np.NaN
    close_time: datetime = np.NaN
    close_price: float = np.NaN
    close_return: float = np.NaN

    def open(self, bin, pair):
        pass

    def _get_return(self):
        """ Calculate return of current position depending on the type of position ie long or short
        """
        return_factor = 1 if self.type == "long" else -1
        return (
            (self.close_price - self.open_price) * return_factor
            / self.open_price
            - self.transaction_cost
        )


    @classmethod
    def reset(cls):
        _position_num = deepcopy( cls.position_num )
        _type = deepcopy( cls.type )
        cls.instance = None        

        cls.instance = Position(position_num = _position_num, type = _type)

    @classmethod
    def close(cls, _bin):
        """ Calculate return of current position and then close it ie reset the position's attributes  """
        # skip min-bins until volume is not zero
        # self._skip_to_non_zero_volume_row(self._position_state["pair"])

        cls.instance.close_price = _bin["middle_median"][ cls.pair ][0]
        cls.instance.close_return = cls.instance._get_return()
        cls.instance.close_time = _bin.index[0]
        cls.instance.duration = pd.Timedelta(cls.open_time - cls.close_time).seconds / 60.0

        try:
            return cls.instance.asdict()
        finally:
            cls.reset()

#%%

def get_total_return(returns_df, costs_bps=[]):
    costs_bps = [ x*0.0005 for x in range(0, 5) ] if not costs_bps else costs_bps
    # get growth rate for all bps
    for cost in costs_bps:
        returns_df[f"return_growth_rate_{ int( cost * 10**4 ) }bps"] = np.where(
            returns_df["type"] == "long",
            (returns_df["close_price"] / returns_df["open_price"]) * (1 - cost) / (1 + cost),
            (returns_df["open_price"] / returns_df["close_price"]) * (1 - cost) / (1 + cost)
        )
    # aggregate over each position and its short/long-positions
    position_returns_df = (
        returns_df
        .groupby( ["position_num", "type"] )
        # multiply each 
        .agg(
            **{ f"total_return_{ int( cost * 10**4 ) }bps": (f"return_growth_rate_{ int( cost * 10**4 ) }bps", "prod") for cost in costs_bps }
        )
        # turn growth rate to 
        - 1
    )
#     # aggregate overeach position's total return
    return position_returns_df.groupby(level="type").agg("mean")

#%%
##### TEST ######
from sklearn.externals import joblib

top10_1min_df = pd.read_csv(
    f"../data/1min/top10_2019_train_test.csv.gz",
    sep=',',
    parse_dates=["time"],
    index_col=['time', 'pair'],
    infer_datetime_format=True,
    compression='gzip',
)
#%%
test_data_df = top10_1min_df[
    (top10_1min_df.index.get_level_values("time") >= "2019-11-01")
    # & (top10_1min_df.index.get_level_values("time") < "2019-11-03") 
]

#%%
# create test and training data
return_colum = "future_return_120min_constraint"
x_columns = [ col for col in  top10_1min_df.columns if "middle_return" in col and "future" not in col ]
x_columns_volume = [ col for col in  top10_1min_df.columns if ("middle_return" in col or "volume_scaled" in col) and "future" not in col ]

column_specs = {
    "no_volume": x_columns,
    # "with_volume": x_columns_volume,
}
# test data from 2019-11-01 to 2019-12-31
# x_test = top10_1min_df[ (top10_1min_df.index.get_level_values("time") >= "2019-11-01") ][x_columns]


#%%
THRESHOLDS = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675]

targets = {
    "future_2state_movement_120min": 120,
    "future_2state_movement_240min": 240,
	# "future_3state_movement_120min": 120,
    # "future_3state_movement_240min": 240,
}

models = ["logistic", "forest"]
models = ["logistic"]
# ["logistic", "forest", ]
for model_name in models:
    for column_spec_name, columns in column_specs.items():
        for target, delta in targets.items():
            print("load data:", model_name, column_spec_name, target)
            x_test = top10_1min_df[ (top10_1min_df.index.get_level_values("time") >= "2019-11-01") ][columns]
            y_test = top10_1min_df[ (top10_1min_df.index.get_level_values("time") >= "2019-11-01") ][target]            
            # get predictions
            clf = joblib.load(f"../models/{model_name}/{model_name}_{column_spec_name}_{target}.pkl")
            predictions = clf.predict_proba( x_test )
            # 
            bt = Backtest(
                test_df=test_data_df,
                prediction_prob=predictions,
                target_column=target, 
                price_column="middle_median",
                return_column=return_colum,
            )
            for threshold in THRESHOLDS:
                print("Threshold:", threshold)
                folder_file_paths = [ x for x in glob.glob1(f"../results/{model_name}/returns/", "*.csv") ]
                file_path = f"../results/{model_name}/returns/trading_returns_{column_spec_name}_{target}_{ round(threshold*100, 1) }_threshold.csv"
                if f"trading_returns_{column_spec_name}_{target}_{ round(threshold*100, 1) }_threshold.csv" not in folder_file_paths:
                    trading_decisions_df = bt.conduct_top_short_long_alg(
                        threshold=threshold,
                        positions=60,
                        skip_zero_volume=True,
                        position_type="both",
                        delta=delta,
                        transaction_cost=0,
                    )
                    trading_decisions_df.to_csv(file_path, index=False)
                    get_total_return(trading_decisions_df).to_csv(
                        f"../results/{model_name}/stats/trading_returns_{column_spec_name}_{target}_{ round(threshold*100, 1) }_threshold.csv"
                    )
                else:
                    print("Skip:", file_path)

#%%

#%%
equal_weight_return_df = bt.get_equal_weight_holding_returns(transaction_cost=0)

equal_weight_return_df.to_csv("../results/equal_weight_hold_returns.csv")


#%%

#%%
