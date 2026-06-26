# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `25`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.642540 | 0.687335 | -0.044795 | 27 | 131 | 0.791139 | 0.841772 |
| active/recent utility | 158 | 1.000 | 0.642540 | 0.687335 | -0.044795 | 27 | 131 | 0.791139 | 0.841772 |
| strong utility action | 89 | 0.563 | 0.630789 | 0.694745 | -0.063956 | 11 | 78 | 0.808989 | 0.910112 |
| utility damage | 33 | 0.209 | 0.586652 | 0.650789 | -0.064137 | 9 | 24 | 0.727273 | 0.909091 |
| active smoke/inferno | 85 | 0.538 | 0.626819 | 0.694504 | -0.067685 | 7 | 78 | 0.800000 | 0.905882 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.642540 | 0.687335 | -0.044795 | 27 | 131 | 0.791139 | 0.841772 |

## Active Smoke/Inferno Intervals

- `9.0s` - `44.0s`, rows `71`
- `69.5s` - `76.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.5483`, XGBoost `0.7138`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5555`, XGBoost `0.7131`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5687`, XGBoost `0.7222`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5640`, XGBoost `0.7129`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5769`, XGBoost `0.7208`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5856`, XGBoost `0.7222`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5774`, XGBoost `0.7129`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3925`, XGBoost `0.5097`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `55.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.8305`, XGBoost `0.7138`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6144`, XGBoost `0.7222`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
