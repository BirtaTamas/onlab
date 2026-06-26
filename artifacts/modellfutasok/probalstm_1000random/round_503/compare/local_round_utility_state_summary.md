# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `22`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.735796 | 0.714004 | 0.021792 | 148 | 82 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.735796 | 0.714004 | 0.021792 | 148 | 82 | 1.000000 | 1.000000 |
| strong utility action | 195 | 0.848 | 0.722974 | 0.702263 | 0.020711 | 129 | 66 | 1.000000 | 1.000000 |
| utility damage | 12 | 0.052 | 0.626252 | 0.598837 | 0.027414 | 12 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 195 | 0.848 | 0.722974 | 0.702263 | 0.020711 | 129 | 66 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.735796 | 0.714004 | 0.021792 | 148 | 82 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `61.0s`, rows `106`
- `62.5s` - `106.5s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `89.0`, LSTM `0.8569`, XGBoost `0.7626`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.8530`, XGBoost `0.7626`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.6175`, XGBoost `0.5280`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6170`, XGBoost `0.5280`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6258`, XGBoost `0.5394`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.8491`, XGBoost `0.7629`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.8561`, XGBoost `0.7734`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.8451`, XGBoost `0.7628`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.8452`, XGBoost `0.7634`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6056`, XGBoost `0.5287`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
