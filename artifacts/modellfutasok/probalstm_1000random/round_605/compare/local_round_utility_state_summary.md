# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `8`
- rows: `113`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 113 | 1.000 | 0.776833 | 0.756756 | 0.020077 | 72 | 41 | 1.000000 | 0.973451 |
| active/recent utility | 113 | 1.000 | 0.776833 | 0.756756 | 0.020077 | 72 | 41 | 1.000000 | 0.973451 |
| strong utility action | 108 | 0.956 | 0.776746 | 0.758346 | 0.018400 | 68 | 40 | 1.000000 | 0.972222 |
| utility damage | 20 | 0.177 | 0.659420 | 0.653927 | 0.005493 | 12 | 8 | 1.000000 | 0.850000 |
| active smoke/inferno | 98 | 0.867 | 0.783125 | 0.770487 | 0.012638 | 58 | 40 | 1.000000 | 0.969388 |
| recent utility last 5s | 10 | 0.088 | 0.714227 | 0.639369 | 0.074859 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 113 | 1.000 | 0.776833 | 0.756756 | 0.020077 | 72 | 41 | 1.000000 | 0.973451 |

## Active Smoke/Inferno Intervals

- `7.0s` - `55.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.8060`, XGBoost `0.6736`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.7787`, XGBoost `0.6743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.7261`, XGBoost `0.6359`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.7226`, XGBoost `0.6359`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `13.5`, LSTM `0.6958`, XGBoost `0.6115`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.7580`, XGBoost `0.6743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.7185`, XGBoost `0.6408`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `26.5`, LSTM `0.6438`, XGBoost `0.5662`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.7095`, XGBoost `0.6323`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6876`, XGBoost `0.6105`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
