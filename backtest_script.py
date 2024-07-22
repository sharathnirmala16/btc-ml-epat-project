import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm.contrib.concurrent import process_map


def fit_model_backtest(kv_pairs: dict) -> tuple[int, dict[str, object]]:
    # getting the parameters from the input dictionary
    dataframe = kv_pairs["dataframe"]
    num_estimators = kv_pairs["num_estimators"]
    freq = kv_pairs["freq"]
    resample = kv_pairs["resample"]

    # creating a copy of the dataframe which will have the indicators applied to it
    df = dataframe[["open", "high", "low", "close"]].copy(deep=True)

    # since I am using 1 minute data, this is not necessary, it's used for resampling OHLC data for higher time intervals
    if resample != "1T":
        ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        df = df.resample(resample).apply(ohlc_dict)

    # indicators applied to the data, a standard 14 period is used for all the indicators, keeping the strategy simple
    # I have used the Random forest algorithm which is basically a collection of decision trees, hence it doesn't require
    # scaling of data. Additionally all the indicators I have used range from 0-100 or -1 to 1 as I take ratios for those that
    # don't have upper and lower bounds, hence the rules to tend to persist for longer periods.
    timeperiod = 14

    df["pct_change"] = df["close"].pct_change()
    df["pct_change_15"] = df["close"].pct_change(15)
    df["rsi"] = talib.RSI(df["close"], timeperiod=timeperiod)
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=timeperiod)
    df["sma"] = talib.SMA(df["close"], timeperiod=timeperiod)
    df["sma/close"] = df["sma"] / df["close"]
    df["corr"] = df["close"].rolling(timeperiod).corr(df["sma"])
    df["volatility"] = df["pct_change"].rolling(timeperiod).std() * 100
    df["volatility_2"] = df["pct_change_15"].rolling(timeperiod).std() * 100
    df["future_pct_change"] = df["pct_change"].shift(-1)
    # df["future_signal"] = np.where(df["future_pct_change"] > 0, 1, 0)
    df["future_signal"] = np.where(
        df["future_pct_change"] > 0, 1, np.where(df["future_pct_change"] < 0, -1, 0)
    )
    df = df.replace([np.inf, -np.inf], np.nan)

    inputs = [
        "pct_change",
        "pct_change_15",
        "rsi",
        "adx",
        "sma/close",
        "corr",
        "volatility",
        "volatility_2",
    ]

    output = "future_signal"

    # using 75% 25% train-test split
    model_window = (3 * df.shape[0]) // 4
    train_data = df.iloc[:model_window].copy(deep=True)
    test_data = df.iloc[model_window:].copy(deep=True)

    # using a Random Forest classifier model to make discrete buy, sell, and hold signals
    model = RandomForestClassifier(
        n_estimators=num_estimators, criterion="gini", random_state=0
    )
    model.fit(train_data[inputs], train_data[output])

    # creating a column for the predictions
    train_data["forecast"] = model.predict(train_data[inputs])
    test_data["forecast"] = model.predict(test_data[inputs])

    # signal is used to buy or sell, essentially the forecast shifted by one to account for the delay in placing the trades
    train_data["signal"] = train_data["forecast"].shift(1)
    test_data["signal"] = test_data["forecast"].shift(1)

    # creating the entries and exits for both the long and short trades.
    test_long_entries = test_data["signal"] == 1
    test_short_entries = test_data["signal"] == -1

    test_long_exits = test_data["signal"].shift(-1) != 1
    test_short_exits = test_data["signal"].shift(-1) != -1

    # using Vectorbt to run a vectorized backtest on the training data
    test_pf = vbt.Portfolio.from_signals(
        test_data["close"],
        entries=test_long_entries,
        exits=test_long_exits,
        short_entries=test_short_entries,
        short_exits=test_short_exits,
        freq=freq,
        size_granularity=1e-8,
    )

    stats = test_pf.stats()

    # tabulating accuracy of the model and other important backtest parameters.
    accuracy = (
        accuracy_score(test_data[output], test_data["forecast"], normalize=True) * 100
    )

    return (
        num_estimators,
        {
            "Sharpe Ratio": round(stats["Sharpe Ratio"], 2),
            "Excess Return [%]": round(
                stats["Total Return [%]"] - stats["Benchmark Return [%]"], 2
            ),
            "Win Rate [%]": round(stats["Win Rate [%]"], 2),
            "Accuracy [%]": round(accuracy, 2),
        },
    )


def main():
    # loading the raw data
    dataframe = pd.read_csv("BTCUSD_M1.csv", index_col=0, parse_dates=True)

    # creating the various key value pairs that will be passed for backtesting
    # the hyperparameter being optimized for is the number of esitmators, i.e the number of decision trees in the model
    kv_pairs_list = [
        {
            "dataframe": dataframe,
            "num_estimators": num_estimators,
            "freq": "1min",
            "resample": "1T",
        }
        for num_estimators in range(5, 26, 1)
    ]

    # using a process pool to run the backtests so that multiple can run in parallel
    # if you are facing CPU temp or RAM usage constraints, consider seting logical=False for the cpu_count
    # or even reduce the number of workers in the max_workers param, this will reduce system resource utilization
    # but your backtests will take longer to complete.
    # for reference, this backtest took me ~6 minutes.
    backtest_results = dict(
        process_map(
            fit_model_backtest,
            kv_pairs_list,
            max_workers=psutil.cpu_count(logical=True),
            desc="Backtests",
        )
    )

    # saving the backtest results
    kpis_df = pd.DataFrame(backtest_results)
    kpis_df.to_csv("KPI_M1.csv")


if __name__ == "__main__":
    main()
