#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from util.report import write_results

#%%
TOP_10_CAPITALIZATION = ['btcusd', 'ethusd', 'eosusd', 'ltcusd', 'xrpusd', 'babusd', 'xmrusd', 'neousd', 'iotusd', "dshusd"]
#%%
def report_results(model, model_name, description, x_test, y_test, plot_per_pair=False):
    test_score = model.score(x_test, y_test)
    print(f"validation score: {model.best_score_}")
    print(f"test score: {test_score}")

    train_start = str(y_train.index[0][0])
    train_end = str(y_train.index[-1][0])
    test_start = str(y_test.index[0][0])
    test_end = str(y_test.index[-1][0])

    write_results(path=f"./results/{model_name}/stats/{model_name}_training_results.txt", 
                entry_text=f"{description}, validation score: {model.best_score_}, test score: {test_score}, training-range: {train_start} to {train_end}, test-range: {test_start} to {test_end}")

    # save figure
    disp = plot_confusion_matrix(
        model, 
        x_test, y_test,   
        # display_labels=["Down", "Up"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.ax_.set_title(f"Confusion Matrix of {model_name} using all pairs\nAccuracy: {round(test_score, 4)}")
    disp.figure_.savefig(f"./results/{model_name}/plots/confusion_matrix_{description}.png")

    if plot_per_pair:
        # create plot for each pair and save stats
        stats_data = {"pair": [], "test_score": []}
        for pair in TOP_10_CAPITALIZATION:
            x_test_ =  x_test[ x_test.index.get_level_values("pair") == pair ]
            y_test_ = y_test[ y_test.index.get_level_values("pair") == pair ]
            test_score_ = model.score(x_test_, y_test_)
            disp = plot_confusion_matrix(
                model, 
                x_test_, y_test_,
                # display_labels=["Down", "Up"],
                cmap=plt.cm.Blues,
                normalize="true",
            )
            # add score to stats_data
            stats_data[ "pair" ].append(pair)
            stats_data[ "test_score" ].append(test_score_)
            
            disp.ax_.set_title(f"Confusion Matrix of {model_name} for {pair}\nAccuracy: {round(test_score_, 4)}")
            disp.figure_.savefig(f"./results/{model_name}/plots/confusion_matrix_{pair}_{description}.png")

        pd.DataFrame.from_dict(stats_data).to_csv(f"./results/{model_name}/stats/{model_name}_stats_by_pair_{description}.csv", index=False)

#%%
print("Load data")
# load 1min-binned data for the top ten pairs
top10_1min_df = pd.read_csv(
    f"./data/1min/top10_2019_train_test.csv.gz",
    sep=',',
    parse_dates=["time"],
    index_col=['time', 'pair'],
    infer_datetime_format=True,
    compression='gzip',
)
#%%
# create training- and test-set
# take only return-columns for training
x_columns = [ col for col in  top10_1min_df.columns if "middle_return" in col and "future" not in col ]
x_columns_volume = [ col for col in  top10_1min_df.columns if ("middle_return" in col or "volume_scaled" in col) and "future" not in col ]
feature_selections = {
    "no_volume": x_columns,
    "with_volume": x_columns_volume,
}



dt = DecisionTreeClassifier(random_state = 0)
models = {
    "forest": RandomForestClassifier(random_state=0, max_features="sqrt"),
    "logistic": LogisticRegression(solver="lbfgs", max_iter=150),
    "adaboost": AdaBoostClassifier(base_estimator=dt),
    "ann": MLPClassifier(random_state=0),
}
search_spaces = {
    "forest": {
        "forest__n_estimators": [100, 500, 1000],
        "forest__max_depth": [3, 5, 10, 15],
    },
    "logistic": {
        "logistic__C": np.logspace(np.log10(0.0001), np.log10(10000), num=100)
    },
    "adaboost": {
        # "adaboost__n_estimators": [50, 100, 200,],
        "adaboost__n_estimators": [500, 1000],
        "adaboost__learning_rate": [0.001, 0.01, 0.1],
        "adaboost__base_estimator__max_depth": [1, 3],
        "adaboost__base_estimator__max_features": [None],
    },
}

#%%
targets = [
    "2state_movement_120min",
    "2state_movement_240min",	
	# "3state_movement_120min",
    # "3state_movement_240min"
]

filtered_1min_df = (
    top10_1min_df
    [ top10_1min_df[ f"volume"].shift(-1) != 0 ]
    .dropna()
)
total_size = filtered_1min_df.shape[0]
#%%
for volume_desc, columns in feature_selections.items():
    for target in targets:
        target = "future_" + target
        print("Target:", target, "Features:", volume_desc)
        # filter out bins for which in the respective next bin no volume was traded and NaN
        # training data from 2019-01-01 to 2019-10-31
        x_train = filtered_1min_df[ (filtered_1min_df.index.get_level_values("time") < "2019-11-01") ][columns]
        y_train = filtered_1min_df[ (filtered_1min_df.index.get_level_values("time") < "2019-11-01") ][target]
        # test data from 2019-11-01 to 2019-12-31
        x_test = filtered_1min_df[ (filtered_1min_df.index.get_level_values("time") >= "2019-11-01") ][columns]
        y_test = filtered_1min_df[ (filtered_1min_df.index.get_level_values("time") >= "2019-11-01") ][target]

        # set validation fold
        non_validation_fold = [ -1 for x in y_train[ (y_train.index.get_level_values("time") < "2019-09-15") ] ] 
        validation_fold = [ 0 for x in y_train[ (y_train.index.get_level_values("time") >= "2019-09-15") ] ]
        ps = PredefinedSplit( non_validation_fold + validation_fold )
        print("Total Size:", total_size,
            "Training Set Size:", x_train.shape,
            "Validation Set Size:", len(validation_fold),
            "Test Set Size:", x_test.shape)
        # ["logistic", "forest"]
        description = f"{volume_desc}_{target}"
        for model_name in ["ann"]:
            filename = f"./models/{model_name}/{model_name}_{description}.pkl"
            print(f"Train {model_name}")


            model = models[ model_name ]
            if model_name != "ann":
                clf_pipline = Pipeline(
                    [("scaler", QuantileTransformer(random_state=0)),
                    (model_name, model)]
                )          
                searchcv = GridSearchCV(
                    clf_pipline,
                    param_grid=search_spaces[ model_name ],
                    cv=ps,
                    n_jobs=6,
                    verbose=10,
                )
                searchcv.fit(x_train, y_train)
                joblib.dump(searchcv, filename)
                report_results(searchcv, model_name, description, x_test, y_test)          
            else:
                search_spaces.update(
                    {
                        "ann": {
                            "hidden_layer_sizes": (x_train.shape[1], x_train.shape[1], 15, 10),
                            "activation": "relu",
                            "solver": "sgd",
                            "alpha": 10**-4,
                            "verbose": 50,
                            "batch_size": 512,
                            "max_iter": 400,
                            "tol": 10**-4,
                            "n_iter_no_change": 400,
                            "random_state": 0,
                        }
                    }
                )
                model = MLPClassifier(**search_spaces["ann"])
                clf_pipline = Pipeline(
                    [("scaler", QuantileTransformer(random_state=0)),
                    ("ann", model)]
                ) 
                clf_pipline.fit(x_train, y_train)
                joblib.dump(clf_pipline, filename)
                # report_results(clf_pipline, model_name, description, x_test, y_test)

              



# %%
