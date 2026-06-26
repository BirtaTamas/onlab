# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `14`
- rows: `278`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 278 | 1.000 | 0.275093 | 0.301356 | -0.026263 | 210 | 68 | 0.758993 | 0.755396 |
| active/recent utility | 278 | 1.000 | 0.275093 | 0.301356 | -0.026263 | 210 | 68 | 0.758993 | 0.755396 |
| strong utility action | 244 | 0.878 | 0.291924 | 0.320101 | -0.028177 | 185 | 59 | 0.725410 | 0.721311 |
| utility damage | 10 | 0.036 | 0.391324 | 0.299955 | 0.091369 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 237 | 0.853 | 0.291827 | 0.321470 | -0.029643 | 184 | 53 | 0.717300 | 0.713080 |
| recent utility last 5s | 10 | 0.036 | 0.281395 | 0.273656 | 0.007739 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 278 | 1.000 | 0.275093 | 0.301356 | -0.026263 | 210 | 68 | 0.758993 | 0.755396 |

## Active Smoke/Inferno Intervals

- `11.5s` - `100.0s`, rows `178`
- `107.0s` - `136.0s`, rows `59`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `108.5`, LSTM `0.0342`, XGBoost `0.2134`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.0383`, XGBoost `0.2134`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.0308`, XGBoost `0.1904`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.4572`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.4567`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.4529`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.4526`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4506`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.4480`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `27.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.4457`, XGBoost `0.3045`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
