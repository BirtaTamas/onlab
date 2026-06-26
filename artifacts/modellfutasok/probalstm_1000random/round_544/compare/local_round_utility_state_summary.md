# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `20`
- rows: `304`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 304 | 1.000 | 0.400067 | 0.369659 | 0.030408 | 102 | 202 | 0.322368 | 0.542763 |
| active/recent utility | 304 | 1.000 | 0.400067 | 0.369659 | 0.030408 | 102 | 202 | 0.322368 | 0.542763 |
| strong utility action | 223 | 0.734 | 0.500254 | 0.462724 | 0.037530 | 43 | 180 | 0.161435 | 0.376682 |
| utility damage | 38 | 0.125 | 0.626089 | 0.562349 | 0.063740 | 1 | 37 | 0.000000 | 0.342105 |
| active smoke/inferno | 223 | 0.734 | 0.500254 | 0.462724 | 0.037530 | 43 | 180 | 0.161435 | 0.376682 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 304 | 1.000 | 0.400067 | 0.369659 | 0.030408 | 102 | 202 | 0.322368 | 0.542763 |

## Active Smoke/Inferno Intervals

- `9.5s` - `120.5s`, rows `223`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `102.5`, LSTM `0.5316`, XGBoost `0.3797`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.0745`, XGBoost `0.2041`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.5851`, XGBoost `0.4572`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5960`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5945`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5942`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5906`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5857`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5851`, XGBoost `0.4682`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.5678`, XGBoost `0.4564`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
