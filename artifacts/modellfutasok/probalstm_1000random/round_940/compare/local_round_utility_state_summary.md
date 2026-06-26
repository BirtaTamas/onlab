# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `10`
- rows: `306`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 306 | 1.000 | 0.281264 | 0.311510 | -0.030245 | 270 | 36 | 0.588235 | 0.529412 |
| active/recent utility | 306 | 1.000 | 0.281264 | 0.311510 | -0.030245 | 270 | 36 | 0.588235 | 0.529412 |
| strong utility action | 243 | 0.794 | 0.328362 | 0.364161 | -0.035799 | 214 | 29 | 0.506173 | 0.456790 |
| utility damage | 12 | 0.039 | 0.528872 | 0.562713 | -0.033841 | 12 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 243 | 0.794 | 0.328362 | 0.364161 | -0.035799 | 214 | 29 | 0.506173 | 0.456790 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 306 | 1.000 | 0.281264 | 0.311510 | -0.030245 | 270 | 36 | 0.588235 | 0.529412 |

## Active Smoke/Inferno Intervals

- `6.0s` - `105.0s`, rows `199`
- `109.0s` - `130.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.5`, LSTM `0.4291`, XGBoost `0.5770`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.6098`, XGBoost `0.7556`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0834`, XGBoost `0.2285`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.4343`, XGBoost `0.5770`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0861`, XGBoost `0.2285`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0861`, XGBoost `0.2285`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.3005`, XGBoost `0.4424`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.2921`, XGBoost `0.4332`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.0885`, XGBoost `0.2285`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4424`, XGBoost `0.5795`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
