# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `26`
- rows: `183`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.724135 | 0.807809 | -0.083674 | 8 | 175 | 1.000000 | 1.000000 |
| active/recent utility | 183 | 1.000 | 0.724135 | 0.807809 | -0.083674 | 8 | 175 | 1.000000 | 1.000000 |
| strong utility action | 127 | 0.694 | 0.701869 | 0.802474 | -0.100605 | 2 | 125 | 1.000000 | 1.000000 |
| utility damage | 12 | 0.066 | 0.616077 | 0.700470 | -0.084393 | 0 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 127 | 0.694 | 0.701869 | 0.802474 | -0.100605 | 2 | 125 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.724135 | 0.807809 | -0.083674 | 8 | 175 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `64.0s`, rows `113`
- `67.5s` - `74.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.6694`, XGBoost `0.8962`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.7218`, XGBoost `0.8945`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7291`, XGBoost `0.8945`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5685`, XGBoost `0.7338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5812`, XGBoost `0.7423`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5837`, XGBoost `0.7418`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7374`, XGBoost `0.8938`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5875`, XGBoost `0.7418`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5884`, XGBoost `0.7418`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5888`, XGBoost `0.7418`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
