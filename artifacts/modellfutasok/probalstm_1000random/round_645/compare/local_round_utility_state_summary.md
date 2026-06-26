# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `6`
- rows: `177`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.743703 | 0.775869 | -0.032166 | 64 | 113 | 1.000000 | 0.988701 |
| active/recent utility | 177 | 1.000 | 0.743703 | 0.775869 | -0.032166 | 64 | 113 | 1.000000 | 0.988701 |
| strong utility action | 158 | 0.893 | 0.746753 | 0.780140 | -0.033387 | 55 | 103 | 1.000000 | 0.987342 |
| utility damage | 30 | 0.169 | 0.650875 | 0.601244 | 0.049632 | 25 | 5 | 1.000000 | 0.933333 |
| active smoke/inferno | 148 | 0.836 | 0.761151 | 0.797515 | -0.036364 | 47 | 101 | 1.000000 | 0.986486 |
| recent utility last 5s | 10 | 0.056 | 0.533674 | 0.523000 | 0.010673 | 8 | 2 | 1.000000 | 1.000000 |
| flash effect present | 177 | 1.000 | 0.743703 | 0.775869 | -0.032166 | 64 | 113 | 1.000000 | 0.988701 |

## Active Smoke/Inferno Intervals

- `9.5s` - `60.0s`, rows `102`
- `64.5s` - `86.0s`, rows `44`
- `87.5s` - `88.0s`, rows `2`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.5306`, XGBoost `0.3389`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `63.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.7884`, XGBoost `0.9273`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.7921`, XGBoost `0.9273`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.7970`, XGBoost `0.9301`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7974`, XGBoost `0.9301`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.8014`, XGBoost `0.9301`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.7976`, XGBoost `0.9217`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.8079`, XGBoost `0.9303`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.8007`, XGBoost `0.9217`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.8066`, XGBoost `0.9273`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
