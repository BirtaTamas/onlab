# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `1`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.179260 | 0.184186 | -0.004927 | 157 | 40 | 0.812183 | 0.842640 |
| active/recent utility | 197 | 1.000 | 0.179260 | 0.184186 | -0.004927 | 157 | 40 | 0.812183 | 0.842640 |
| strong utility action | 68 | 0.345 | 0.186056 | 0.174852 | 0.011204 | 40 | 28 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 68 | 0.345 | 0.186056 | 0.174852 | 0.011204 | 40 | 28 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.179260 | 0.184186 | -0.004927 | 157 | 40 | 0.812183 | 0.842640 |

## Active Smoke/Inferno Intervals

- `26.5s` - `60.0s`, rows `68`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.5`, LSTM `0.4539`, XGBoost `0.2920`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4371`, XGBoost `0.2920`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.4356`, XGBoost `0.2920`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4328`, XGBoost `0.2920`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4248`, XGBoost `0.2920`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4134`, XGBoost `0.2920`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1308`, XGBoost `0.2387`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.3979`, XGBoost `0.2920`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1376`, XGBoost `0.2387`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3917`, XGBoost `0.2920`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
