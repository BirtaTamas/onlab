# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `2`
- rows: `102`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 102 | 1.000 | 0.018340 | 0.031274 | -0.012934 | 95 | 7 | 1.000000 | 1.000000 |
| active/recent utility | 102 | 1.000 | 0.018340 | 0.031274 | -0.012934 | 95 | 7 | 1.000000 | 1.000000 |
| strong utility action | 82 | 0.804 | 0.018672 | 0.031427 | -0.012755 | 76 | 6 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 82 | 0.804 | 0.018672 | 0.031427 | -0.012755 | 76 | 6 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 102 | 1.000 | 0.018340 | 0.031274 | -0.012934 | 95 | 7 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `50.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.0274`, XGBoost `0.0997`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0097`, XGBoost `0.0483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0129`, XGBoost `0.0462`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `151.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0139`, XGBoost `0.0426`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `143.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0219`, XGBoost `0.0493`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0223`, XGBoost `0.0493`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0129`, XGBoost `0.0381`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `77.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0130`, XGBoost `0.0380`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `110.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0245`, XGBoost `0.0490`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0130`, XGBoost `0.0342`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `151.0`, recent_utility `0`
