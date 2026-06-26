# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `3`
- rows: `124`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 124 | 1.000 | 0.830331 | 0.818294 | 0.012037 | 66 | 58 | 1.000000 | 1.000000 |
| active/recent utility | 124 | 1.000 | 0.830331 | 0.818294 | 0.012037 | 66 | 58 | 1.000000 | 1.000000 |
| strong utility action | 77 | 0.621 | 0.814559 | 0.799076 | 0.015484 | 46 | 31 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 77 | 0.621 | 0.814559 | 0.799076 | 0.015484 | 46 | 31 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 124 | 1.000 | 0.830331 | 0.818294 | 0.012037 | 66 | 58 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `48.0s`, rows `77`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.7639`, XGBoost `0.6887`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7637`, XGBoost `0.6893`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7567`, XGBoost `0.6844`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.7539`, XGBoost `0.6886`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.7507`, XGBoost `0.6859`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.7519`, XGBoost `0.6886`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.7470`, XGBoost `0.6845`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.7510`, XGBoost `0.6886`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.7476`, XGBoost `0.6859`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.7495`, XGBoost `0.6893`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
