# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `11`
- rows: `103`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 103 | 1.000 | 0.696504 | 0.799330 | -0.102826 | 0 | 103 | 1.000000 | 1.000000 |
| active/recent utility | 103 | 1.000 | 0.696504 | 0.799330 | -0.102826 | 0 | 103 | 1.000000 | 1.000000 |
| strong utility action | 87 | 0.845 | 0.719699 | 0.836540 | -0.116841 | 0 | 87 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 87 | 0.845 | 0.719699 | 0.836540 | -0.116841 | 0 | 87 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.107 | 0.685336 | 0.841336 | -0.156000 | 0 | 11 | 1.000000 | 1.000000 |
| flash effect present | 103 | 1.000 | 0.696504 | 0.799330 | -0.102826 | 0 | 103 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.5788`, XGBoost `0.8139`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5812`, XGBoost `0.8135`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5794`, XGBoost `0.8112`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5843`, XGBoost `0.8124`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5849`, XGBoost `0.8118`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5875`, XGBoost `0.8054`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5918`, XGBoost `0.8089`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `31.5`, LSTM `0.5969`, XGBoost `0.8134`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5952`, XGBoost `0.8112`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5954`, XGBoost `0.8089`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
