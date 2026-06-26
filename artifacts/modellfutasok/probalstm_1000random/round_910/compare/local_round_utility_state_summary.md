# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `12`
- rows: `129`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 129 | 1.000 | 0.269234 | 0.178219 | 0.091016 | 4 | 125 | 0.767442 | 0.829457 |
| active/recent utility | 129 | 1.000 | 0.269234 | 0.178219 | 0.091016 | 4 | 125 | 0.767442 | 0.829457 |
| strong utility action | 104 | 0.806 | 0.212223 | 0.115414 | 0.096808 | 4 | 100 | 0.894231 | 0.971154 |
| utility damage | 10 | 0.078 | 0.089683 | 0.053836 | 0.035847 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 104 | 0.806 | 0.212223 | 0.115414 | 0.096808 | 4 | 100 | 0.894231 | 0.971154 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 129 | 1.000 | 0.269234 | 0.178219 | 0.091016 | 4 | 125 | 0.767442 | 0.829457 |

## Active Smoke/Inferno Intervals

- `9.5s` - `61.0s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.5494`, XGBoost `0.2396`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5515`, XGBoost `0.2435`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5511`, XGBoost `0.2435`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.3740`, XGBoost `0.0700`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5410`, XGBoost `0.2389`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5369`, XGBoost `0.2359`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5247`, XGBoost `0.2363`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5143`, XGBoost `0.2359`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5048`, XGBoost `0.2348`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.4965`, XGBoost `0.2329`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
