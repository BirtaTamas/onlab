# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `8`
- rows: `225`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.349251 | 0.390837 | -0.041586 | 224 | 1 | 0.684444 | 0.453333 |
| active/recent utility | 225 | 1.000 | 0.349251 | 0.390837 | -0.041586 | 224 | 1 | 0.684444 | 0.453333 |
| strong utility action | 160 | 0.711 | 0.375633 | 0.424942 | -0.049309 | 160 | 0 | 0.681250 | 0.362500 |
| utility damage | 10 | 0.044 | 0.477196 | 0.534807 | -0.057611 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 150 | 0.667 | 0.367447 | 0.417237 | -0.049791 | 150 | 0 | 0.693333 | 0.386667 |
| recent utility last 5s | 10 | 0.044 | 0.498435 | 0.540517 | -0.042082 | 10 | 0 | 0.500000 | 0.000000 |
| flash effect present | 225 | 1.000 | 0.349251 | 0.390837 | -0.041586 | 224 | 1 | 0.684444 | 0.453333 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.0s`, rows `92`
- `64.0s` - `92.5s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.6296`, XGBoost `0.7773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6535`, XGBoost `0.7773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6584`, XGBoost `0.7773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.6591`, XGBoost `0.7773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4418`, XGBoost `0.5573`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4511`, XGBoost `0.5586`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.0238`, XGBoost `0.1296`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.4807`, XGBoost `0.5788`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4515`, XGBoost `0.5496`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.8175`, XGBoost `0.9099`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
