# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `5`
- rows: `224`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.435058 | 0.450561 | -0.015503 | 141 | 83 | 0.428571 | 0.428571 |
| active/recent utility | 224 | 1.000 | 0.435058 | 0.450561 | -0.015503 | 141 | 83 | 0.428571 | 0.428571 |
| strong utility action | 161 | 0.719 | 0.547726 | 0.563592 | -0.015866 | 90 | 71 | 0.279503 | 0.279503 |
| utility damage | 20 | 0.089 | 0.722018 | 0.733568 | -0.011550 | 11 | 9 | 0.000000 | 0.000000 |
| active smoke/inferno | 161 | 0.719 | 0.547726 | 0.563592 | -0.015866 | 90 | 71 | 0.279503 | 0.279503 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.435058 | 0.450561 | -0.015503 | 141 | 83 | 0.428571 | 0.428571 |

## Active Smoke/Inferno Intervals

- `6.0s` - `86.0s`, rows `161`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.0`, LSTM `0.5401`, XGBoost `0.7267`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5488`, XGBoost `0.7243`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5513`, XGBoost `0.7262`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5892`, XGBoost `0.7206`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5740`, XGBoost `0.6919`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5739`, XGBoost `0.6862`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6144`, XGBoost `0.7229`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.2329`, XGBoost `0.3344`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6189`, XGBoost `0.7185`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6352`, XGBoost `0.7331`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
