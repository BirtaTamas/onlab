# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `9`
- rows: `182`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.499247 | 0.262264 | 0.714535 | 0.368132 | 0.500753 |
| xgboost | 0.479012 | 0.243609 | 0.672053 | 0.390110 | 0.520988 |

## Closer Per Tick

- lstm: `82`
- xgboost: `100`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
