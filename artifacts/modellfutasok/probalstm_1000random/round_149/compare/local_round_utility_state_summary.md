# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `1`
- rows: `137`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.281623 | 0.379826 | -0.098202 | 124 | 13 | 0.934307 | 0.978102 |
| active/recent utility | 137 | 1.000 | 0.281623 | 0.379826 | -0.098202 | 124 | 13 | 0.934307 | 0.978102 |
| strong utility action | 54 | 0.394 | 0.156916 | 0.326725 | -0.169808 | 54 | 0 | 1.000000 | 0.944444 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 54 | 0.394 | 0.156916 | 0.326725 | -0.169808 | 54 | 0 | 1.000000 | 0.944444 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 137 | 1.000 | 0.281623 | 0.379826 | -0.098202 | 124 | 13 | 0.934307 | 0.978102 |

## Active Smoke/Inferno Intervals

- `35.0s` - `61.5s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.5`, LSTM `0.0810`, XGBoost `0.4065`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0914`, XGBoost `0.4098`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0847`, XGBoost `0.4018`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0844`, XGBoost `0.3994`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0905`, XGBoost `0.4010`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0663`, XGBoost `0.3726`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1049`, XGBoost `0.3973`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0288`, XGBoost `0.3147`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0650`, XGBoost `0.3482`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0321`, XGBoost `0.3126`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
