# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `2`
- rows: `132`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.874009 | 0.983185 | -0.109176 | 0 | 132 | 1.000000 | 1.000000 |
| active/recent utility | 97 | 0.735 | 0.875401 | 0.983690 | -0.108289 | 0 | 97 | 1.000000 | 1.000000 |
| strong utility action | 64 | 0.485 | 0.861901 | 0.983140 | -0.121239 | 0 | 64 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.076 | 0.962413 | 0.991587 | -0.029174 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 54 | 0.409 | 0.843288 | 0.981576 | -0.138288 | 0 | 54 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 43 | 0.326 | 0.915729 | 0.986345 | -0.070615 | 0 | 43 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `33.5s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.7810`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7878`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7883`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7886`, XGBoost `0.9819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7912`, XGBoost `0.9819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.7920`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.8008`, XGBoost `0.9819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.8008`, XGBoost `0.9819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.8023`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.8031`, XGBoost `0.9819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
