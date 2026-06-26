# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m1-dust2.csv`
- round_num: `5`
- rows: `287`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 287 | 1.000 | 0.512504 | 0.488530 | 0.023974 | 204 | 83 | 0.620209 | 0.337979 |
| active/recent utility | 287 | 1.000 | 0.512504 | 0.488530 | 0.023974 | 204 | 83 | 0.620209 | 0.337979 |
| strong utility action | 220 | 0.767 | 0.536666 | 0.508057 | 0.028609 | 165 | 55 | 0.668182 | 0.340909 |
| utility damage | 10 | 0.035 | 0.513799 | 0.493267 | 0.020532 | 9 | 1 | 0.700000 | 0.000000 |
| active smoke/inferno | 220 | 0.767 | 0.536666 | 0.508057 | 0.028609 | 165 | 55 | 0.668182 | 0.340909 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 287 | 1.000 | 0.512504 | 0.488530 | 0.023974 | 204 | 83 | 0.620209 | 0.337979 |

## Active Smoke/Inferno Intervals

- `6.0s` - `115.5s`, rows `220`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `101.5`, LSTM `0.1590`, XGBoost `0.4576`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.1343`, XGBoost `0.4291`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.4010`, XGBoost `0.1447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.3822`, XGBoost `0.1436`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.3804`, XGBoost `0.1428`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.3557`, XGBoost `0.1262`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.3951`, XGBoost `0.1783`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.2164`, XGBoost `0.4298`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.3969`, XGBoost `0.1846`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.3209`, XGBoost `0.1283`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
