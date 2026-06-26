# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `2`
- rows: `137`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.257299 | 0.251372 | 0.005927 | 62 | 75 | 1.000000 | 1.000000 |
| active/recent utility | 137 | 1.000 | 0.257299 | 0.251372 | 0.005927 | 62 | 75 | 1.000000 | 1.000000 |
| strong utility action | 124 | 0.905 | 0.258003 | 0.251919 | 0.006084 | 56 | 68 | 1.000000 | 1.000000 |
| utility damage | 19 | 0.139 | 0.154850 | 0.181364 | -0.026514 | 16 | 3 | 1.000000 | 1.000000 |
| active smoke/inferno | 124 | 0.905 | 0.258003 | 0.251919 | 0.006084 | 56 | 68 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 137 | 1.000 | 0.257299 | 0.251372 | 0.005927 | 62 | 75 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `68.0s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.1606`, XGBoost `0.3499`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.4619`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4490`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.4484`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.4465`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1510`, XGBoost `0.2977`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4247`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4199`, XGBoost `0.2982`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4129`, XGBoost `0.3005`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1697`, XGBoost `0.2774`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
