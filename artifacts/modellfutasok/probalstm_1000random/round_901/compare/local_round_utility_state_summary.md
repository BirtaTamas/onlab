# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `9`
- rows: `223`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.264061 | 0.250215 | 0.013846 | 99 | 124 | 0.771300 | 0.798206 |
| active/recent utility | 223 | 1.000 | 0.264061 | 0.250215 | 0.013846 | 99 | 124 | 0.771300 | 0.798206 |
| strong utility action | 165 | 0.740 | 0.331002 | 0.306592 | 0.024409 | 46 | 119 | 0.709091 | 0.745455 |
| utility damage | 10 | 0.045 | 0.616054 | 0.574567 | 0.041487 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 156 | 0.700 | 0.318501 | 0.294991 | 0.023511 | 46 | 110 | 0.750000 | 0.788462 |
| recent utility last 5s | 10 | 0.045 | 0.546201 | 0.507407 | 0.038794 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 223 | 1.000 | 0.264061 | 0.250215 | 0.013846 | 99 | 124 | 0.771300 | 0.798206 |

## Active Smoke/Inferno Intervals

- `6.0s` - `54.5s`, rows `98`
- `58.5s` - `87.0s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.0`, LSTM `0.4961`, XGBoost `0.2899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4823`, XGBoost `0.2899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4790`, XGBoost `0.2899`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4743`, XGBoost `0.2899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4716`, XGBoost `0.2901`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.4702`, XGBoost `0.2899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4664`, XGBoost `0.2901`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4479`, XGBoost `0.2901`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4406`, XGBoost `0.2901`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4353`, XGBoost `0.2895`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
