import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm.contrib.concurrent import process_map


def fit_model_backtest(kv_pairs: dict) -> tuple[int, dict[str, object]]:
    dataframe = kv_pairs["dataframe"]
    num_estimators = kv_pairs["num_estimators"]
    freq = kv_pairs["freq"]
    resample = kv_pairs["resample"]

    df = dataframe[["open", "high", "low", "close"]].copy(deep=True)

    if resample != "1T":
        ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        df = df.resample(resample).apply(ohlc_dict)

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

    model_window = (3 * df.shape[0]) // 4
    train_data = df.iloc[:model_window].copy(deep=True)
    test_data = df.iloc[model_window:].copy(deep=True)

    model = RandomForestClassifier(
        n_estimators=num_estimators, criterion="gini", random_state=0
    )
    model.fit(train_data[inputs], train_data[output])

    train_data["forecast"] = model.predict(train_data[inputs])
    test_data["forecast"] = model.predict(test_data[inputs])
    train_data["signal"] = train_data["forecast"].shift(1)
    test_data["signal"] = test_data["forecast"].shift(1)

    test_long_entries = test_data["signal"] == 1
    test_short_entries = test_data["signal"] == -1

    test_long_exits = test_data["signal"].shift(-1) != 1
    test_short_exits = test_data["signal"].shift(-1) != -1

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
    dataframe = pd.read_csv("BTCUSD_M1.csv", index_col=0, parse_dates=True)

    kv_pairs_list = [
        {
            "dataframe": dataframe,
            "num_estimators": num_estimators,
            "freq": "1min",
            "resample": "1T",
        }
        for num_estimators in range(5, 26, 1)
    ]

    backtest_results = dict(
        process_map(
            fit_model_backtest,
            kv_pairs_list,
            max_workers=psutil.cpu_count(logical=True),
            desc="Backtests",
        )
    )

    kpis_df = pd.DataFrame(backtest_results)
    kpis_df.to_csv("KPI_M1.csv")


if __name__ == "__main__":
    main()
