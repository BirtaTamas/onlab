# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `11`
- rows: `200`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.662219 | 0.681354 | -0.019135 | 141 | 59 | 0.010000 | 0.020000 |
| active/recent utility | 200 | 1.000 | 0.662219 | 0.681354 | -0.019135 | 141 | 59 | 0.010000 | 0.020000 |
| strong utility action | 160 | 0.800 | 0.624540 | 0.640288 | -0.015749 | 109 | 51 | 0.000000 | 0.025000 |
| utility damage | 10 | 0.050 | 0.667284 | 0.672290 | -0.005006 | 6 | 4 | 0.000000 | 0.000000 |
| active smoke/inferno | 153 | 0.765 | 0.622483 | 0.643811 | -0.021327 | 109 | 44 | 0.000000 | 0.026144 |
| recent utility last 5s | 20 | 0.100 | 0.647733 | 0.601832 | 0.045900 | 9 | 11 | 0.000000 | 0.000000 |
| flash effect present | 200 | 1.000 | 0.662219 | 0.681354 | -0.019135 | 141 | 59 | 0.010000 | 0.020000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `83.0s`, rows `153`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.8058`, XGBoost `0.9494`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.7191`, XGBoost `0.5878`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.8179`, XGBoost `0.9489`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6686`, XGBoost `0.7979`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6796`, XGBoost `0.5593`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.5225`, XGBoost `0.6408`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5862`, XGBoost `0.7041`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.8312`, XGBoost `0.9484`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.6731`, XGBoost `0.5562`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `57.0`, LSTM `0.5235`, XGBoost `0.6400`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
