# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `14`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.671944 | 0.748476 | -0.076532 | 0 | 202 | 1.000000 | 1.000000 |
| active/recent utility | 202 | 1.000 | 0.671944 | 0.748476 | -0.076532 | 0 | 202 | 1.000000 | 1.000000 |
| strong utility action | 177 | 0.876 | 0.675492 | 0.755666 | -0.080175 | 0 | 177 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.050 | 0.594636 | 0.637151 | -0.042516 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 177 | 0.876 | 0.675492 | 0.755666 | -0.080175 | 0 | 177 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 202 | 1.000 | 0.671944 | 0.748476 | -0.076532 | 0 | 202 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `57.0s`, rows `99`
- `62.0s` - `100.5s`, rows `78`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.6146`, XGBoost `0.8056`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.7256`, XGBoost `0.9001`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.6280`, XGBoost `0.8018`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.7234`, XGBoost `0.8966`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.7287`, XGBoost `0.8998`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.7284`, XGBoost `0.8964`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6407`, XGBoost `0.8064`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6350`, XGBoost `0.8004`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6419`, XGBoost `0.8058`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6378`, XGBoost `0.8004`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
