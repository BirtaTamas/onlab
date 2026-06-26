# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `5`
- rows: `119`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 119 | 1.000 | 0.682332 | 0.754323 | -0.071991 | 4 | 115 | 0.689076 | 1.000000 |
| active/recent utility | 119 | 1.000 | 0.682332 | 0.754323 | -0.071991 | 4 | 115 | 0.689076 | 1.000000 |
| strong utility action | 110 | 0.924 | 0.699705 | 0.771753 | -0.072048 | 4 | 106 | 0.745455 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 106 | 0.891 | 0.708672 | 0.780745 | -0.072073 | 4 | 102 | 0.773585 | 1.000000 |
| recent utility last 5s | 10 | 0.084 | 0.459486 | 0.537275 | -0.077789 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 119 | 1.000 | 0.682332 | 0.754323 | -0.071991 | 4 | 115 | 0.689076 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.4139`, XGBoost `0.6844`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.4649`, XGBoost `0.7200`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.4324`, XGBoost `0.6837`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6312`, XGBoost `0.8537`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4909`, XGBoost `0.6832`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.4999`, XGBoost `0.6832`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.6803`, XGBoost `0.8431`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5179`, XGBoost `0.6774`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6968`, XGBoost `0.8538`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5280`, XGBoost `0.6804`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
