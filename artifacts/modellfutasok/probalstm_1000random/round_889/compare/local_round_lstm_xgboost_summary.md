# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `10`
- rows: `200`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.195950 | 0.058107 | 0.235462 | 0.985000 | 0.804050 |
| xgboost | 0.290126 | 0.122422 | 0.381801 | 0.980000 | 0.709874 |

## Closer Per Tick

- lstm: `169`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
