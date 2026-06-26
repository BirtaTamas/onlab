# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `11`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.574573 | 0.624567 | -0.049994 | 53 | 145 | 0.656566 | 0.873737 |
| active/recent utility | 198 | 1.000 | 0.574573 | 0.624567 | -0.049994 | 53 | 145 | 0.656566 | 0.873737 |
| strong utility action | 124 | 0.626 | 0.549595 | 0.616032 | -0.066437 | 23 | 101 | 0.637097 | 0.862903 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.626 | 0.549595 | 0.616032 | -0.066437 | 23 | 101 | 0.637097 | 0.862903 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.574573 | 0.624567 | -0.049994 | 53 | 145 | 0.656566 | 0.873737 |

## Active Smoke/Inferno Intervals

- `6.5s` - `37.5s`, rows `63`
- `63.5s` - `93.5s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `84.0`, LSTM `0.2764`, XGBoost `0.5007`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.3600`, XGBoost `0.5441`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.3144`, XGBoost `0.4961`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.4384`, XGBoost `0.6121`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.3250`, XGBoost `0.4981`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.3430`, XGBoost `0.5047`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.5422`, XGBoost `0.7025`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `57.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5397`, XGBoost `0.6994`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `19.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.3512`, XGBoost `0.5039`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2764`, XGBoost `0.4263`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
