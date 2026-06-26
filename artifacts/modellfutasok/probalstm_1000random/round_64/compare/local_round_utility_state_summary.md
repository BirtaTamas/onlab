# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `11`
- rows: `151`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.494125 | 0.521347 | -0.027221 | 85 | 66 | 0.476821 | 0.423841 |
| active/recent utility | 151 | 1.000 | 0.494125 | 0.521347 | -0.027221 | 85 | 66 | 0.476821 | 0.423841 |
| strong utility action | 150 | 0.993 | 0.492735 | 0.520814 | -0.028080 | 85 | 65 | 0.480000 | 0.426667 |
| utility damage | 30 | 0.199 | 0.600203 | 0.624779 | -0.024576 | 19 | 11 | 0.200000 | 0.066667 |
| active smoke/inferno | 138 | 0.914 | 0.472970 | 0.512696 | -0.039726 | 85 | 53 | 0.521739 | 0.463768 |
| recent utility last 5s | 22 | 0.146 | 0.687461 | 0.623192 | 0.064269 | 3 | 19 | 0.000000 | 0.000000 |
| flash effect present | 151 | 1.000 | 0.494125 | 0.521347 | -0.027221 | 85 | 66 | 0.476821 | 0.423841 |

## Active Smoke/Inferno Intervals

- `6.5s` - `75.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.2346`, XGBoost `0.5574`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.2718`, XGBoost `0.5587`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.3264`, XGBoost `0.5574`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4973`, XGBoost `0.7017`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3884`, XGBoost `0.5585`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.2787`, XGBoost `0.4391`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.2938`, XGBoost `0.4372`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2947`, XGBoost `0.4372`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2936`, XGBoost `0.4357`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.4225`, XGBoost `0.5576`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `48.0`, recent_utility `0`
