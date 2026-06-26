# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `9`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.571571 | 0.679192 | -0.107621 | 0 | 230 | 0.756522 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.571571 | 0.679192 | -0.107621 | 0 | 230 | 0.756522 | 1.000000 |
| strong utility action | 207 | 0.900 | 0.588957 | 0.694560 | -0.105603 | 0 | 207 | 0.787440 | 1.000000 |
| utility damage | 12 | 0.052 | 0.504168 | 0.565983 | -0.061814 | 0 | 12 | 0.833333 | 1.000000 |
| active smoke/inferno | 207 | 0.900 | 0.588957 | 0.694560 | -0.105603 | 0 | 207 | 0.787440 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.571571 | 0.679192 | -0.107621 | 0 | 230 | 0.756522 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `109.5s`, rows `207`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `109.0`, LSTM `0.1796`, XGBoost `0.5299`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.1828`, XGBoost `0.5299`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.1983`, XGBoost `0.5381`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.1951`, XGBoost `0.5304`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.2161`, XGBoost `0.5357`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.2269`, XGBoost `0.5373`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.2936`, XGBoost `0.5298`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.3033`, XGBoost `0.5309`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.3379`, XGBoost `0.5269`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.5586`, XGBoost `0.7311`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
