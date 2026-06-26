# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `13`
- rows: `114`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 114 | 1.000 | 0.643350 | 0.672997 | -0.029647 | 30 | 84 | 0.885965 | 0.947368 |
| active/recent utility | 114 | 1.000 | 0.643350 | 0.672997 | -0.029647 | 30 | 84 | 0.885965 | 0.947368 |
| strong utility action | 92 | 0.807 | 0.671633 | 0.706451 | -0.034818 | 13 | 79 | 0.880435 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 92 | 0.807 | 0.671633 | 0.706451 | -0.034818 | 13 | 79 | 0.880435 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 114 | 1.000 | 0.643350 | 0.672997 | -0.029647 | 30 | 84 | 0.885965 | 0.947368 |

## Active Smoke/Inferno Intervals

- `6.5s` - `31.5s`, rows `51`
- `36.5s` - `56.5s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.6045`, XGBoost `0.7786`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6135`, XGBoost `0.7786`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7950`, XGBoost `0.9520`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5927`, XGBoost `0.7357`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5996`, XGBoost `0.7357`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.8379`, XGBoost `0.9487`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8652`, XGBoost `0.9603`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.8668`, XGBoost `0.9603`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.8706`, XGBoost `0.9597`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.8714`, XGBoost `0.9603`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
