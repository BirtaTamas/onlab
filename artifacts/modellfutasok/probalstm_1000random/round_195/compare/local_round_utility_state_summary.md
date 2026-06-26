# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `4`
- rows: `274`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 274 | 1.000 | 0.342198 | 0.388765 | -0.046567 | 263 | 11 | 0.503650 | 0.463504 |
| active/recent utility | 274 | 1.000 | 0.342198 | 0.388765 | -0.046567 | 263 | 11 | 0.503650 | 0.463504 |
| strong utility action | 179 | 0.653 | 0.421531 | 0.482814 | -0.061283 | 168 | 11 | 0.329609 | 0.290503 |
| utility damage | 23 | 0.084 | 0.560576 | 0.613912 | -0.053336 | 23 | 0 | 0.217391 | 0.000000 |
| active smoke/inferno | 179 | 0.653 | 0.421531 | 0.482814 | -0.061283 | 168 | 11 | 0.329609 | 0.290503 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 274 | 1.000 | 0.342198 | 0.388765 | -0.046567 | 263 | 11 | 0.503650 | 0.463504 |

## Active Smoke/Inferno Intervals

- `9.5s` - `56.5s`, rows `95`
- `57.5s` - `68.0s`, rows `22`
- `76.0s` - `106.5s`, rows `62`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `85.0`, LSTM `0.1071`, XGBoost `0.3284`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.0778`, XGBoost `0.2977`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.0656`, XGBoost `0.2835`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0344`, XGBoost `0.2174`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.0358`, XGBoost `0.2164`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.0236`, XGBoost `0.1790`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.0256`, XGBoost `0.1782`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.0214`, XGBoost `0.1734`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.0528`, XGBoost `0.2013`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.0557`, XGBoost `0.2018`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
