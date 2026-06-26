# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `10`
- rows: `167`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.762178 | 0.774128 | -0.011950 | 47 | 120 | 1.000000 | 1.000000 |
| active/recent utility | 167 | 1.000 | 0.762178 | 0.774128 | -0.011950 | 47 | 120 | 1.000000 | 1.000000 |
| strong utility action | 163 | 0.976 | 0.763018 | 0.775053 | -0.012035 | 45 | 118 | 1.000000 | 1.000000 |
| utility damage | 17 | 0.102 | 0.827861 | 0.834498 | -0.006637 | 5 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 148 | 0.886 | 0.767069 | 0.779396 | -0.012327 | 40 | 108 | 1.000000 | 1.000000 |
| recent utility last 5s | 31 | 0.186 | 0.725644 | 0.726852 | -0.001207 | 17 | 14 | 1.000000 | 1.000000 |
| flash effect present | 167 | 1.000 | 0.762178 | 0.774128 | -0.011950 | 47 | 120 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `83.0s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.6698`, XGBoost `0.7214`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6746`, XGBoost `0.7214`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6747`, XGBoost `0.7203`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6763`, XGBoost `0.7205`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6929`, XGBoost `0.7335`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6925`, XGBoost `0.7329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6931`, XGBoost `0.7329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6826`, XGBoost `0.7210`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6831`, XGBoost `0.7210`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6951`, XGBoost `0.7329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
