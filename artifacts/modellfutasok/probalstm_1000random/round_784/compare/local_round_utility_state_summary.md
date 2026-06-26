# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `1`
- rows: `188`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.627639 | 0.751770 | -0.124131 | 41 | 147 | 0.845745 | 0.808511 |
| active/recent utility | 162 | 0.862 | 0.647376 | 0.791885 | -0.144509 | 25 | 137 | 0.901235 | 0.845679 |
| strong utility action | 61 | 0.324 | 0.596391 | 0.667664 | -0.071273 | 25 | 36 | 0.868852 | 0.590164 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 61 | 0.324 | 0.596391 | 0.667664 | -0.071273 | 25 | 36 | 0.868852 | 0.590164 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 0.766 | 0.662743 | 0.830076 | -0.167333 | 7 | 137 | 0.888889 | 0.951389 |

## Active Smoke/Inferno Intervals

- `13.0s` - `43.0s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.0`, LSTM `0.6208`, XGBoost `0.8397`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6567`, XGBoost `0.8639`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5385`, XGBoost `0.7453`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.6572`, XGBoost `0.8639`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.6633`, XGBoost `0.8639`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6877`, XGBoost `0.8855`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5559`, XGBoost `0.7524`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.6744`, XGBoost `0.8639`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6512`, XGBoost `0.8397`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6755`, XGBoost `0.8639`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
