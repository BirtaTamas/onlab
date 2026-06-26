# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `13`
- rows: `194`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.229878 | 0.264859 | -0.034981 | 186 | 8 | 0.860825 | 0.659794 |
| active/recent utility | 194 | 1.000 | 0.229878 | 0.264859 | -0.034981 | 186 | 8 | 0.860825 | 0.659794 |
| strong utility action | 62 | 0.320 | 0.159576 | 0.194828 | -0.035252 | 54 | 8 | 0.822581 | 0.903226 |
| utility damage | 10 | 0.052 | 0.505934 | 0.502252 | 0.003682 | 2 | 8 | 0.000000 | 0.800000 |
| active smoke/inferno | 62 | 0.320 | 0.159576 | 0.194828 | -0.035252 | 54 | 8 | 0.822581 | 0.903226 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 194 | 1.000 | 0.229878 | 0.264859 | -0.034981 | 186 | 8 | 0.860825 | 0.659794 |

## Active Smoke/Inferno Intervals

- `30.0s` - `35.0s`, rows `11`
- `41.5s` - `66.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.0523`, XGBoost `0.2470`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0857`, XGBoost `0.2487`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0343`, XGBoost `0.1333`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.3961`, XGBoost `0.4945`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.4061`, XGBoost `0.4979`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.4713`, XGBoost `0.5623`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4155`, XGBoost `0.5011`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.4131`, XGBoost `0.4939`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.4136`, XGBoost `0.4939`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3728`, XGBoost `0.4524`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
