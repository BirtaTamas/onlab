# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `4`
- rows: `178`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.186566 | 0.078126 | 0.243734 | 0.977528 | 0.186566 |
| xgboost | 0.245263 | 0.102955 | 0.322006 | 0.870787 | 0.245263 |

## Closer Per Tick

- lstm: `161`
- xgboost: `17`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
