# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `12`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.599926 | 0.402746 | 0.197180 | 158 | 12 | 0.735294 | 0.429412 |
| active/recent utility | 170 | 1.000 | 0.599926 | 0.402746 | 0.197180 | 158 | 12 | 0.735294 | 0.429412 |
| strong utility action | 144 | 0.847 | 0.588644 | 0.383864 | 0.204780 | 132 | 12 | 0.729167 | 0.368056 |
| utility damage | 41 | 0.241 | 0.597636 | 0.366170 | 0.231466 | 41 | 0 | 0.731707 | 0.292683 |
| active smoke/inferno | 144 | 0.847 | 0.588644 | 0.383864 | 0.204780 | 132 | 12 | 0.729167 | 0.368056 |
| recent utility last 5s | 20 | 0.118 | 0.607961 | 0.337042 | 0.270919 | 20 | 0 | 1.000000 | 0.400000 |
| flash effect present | 170 | 1.000 | 0.599926 | 0.402746 | 0.197180 | 158 | 12 | 0.735294 | 0.429412 |

## Active Smoke/Inferno Intervals

- `10.0s` - `61.5s`, rows `104`
- `65.0s` - `84.5s`, rows `40`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.5241`, XGBoost `0.1444`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `53.0`, recent_utility `1`
- seconds `51.0`, LSTM `0.5200`, XGBoost `0.1430`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `1`
- seconds `52.0`, LSTM `0.5095`, XGBoost `0.1430`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `53.0`, recent_utility `1`
- seconds `52.5`, LSTM `0.5248`, XGBoost `0.1593`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `53.0`, recent_utility `1`
- seconds `50.5`, LSTM `0.5086`, XGBoost `0.1437`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `1`
- seconds `50.0`, LSTM `0.5041`, XGBoost `0.1414`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `53.0`, recent_utility `1`
- seconds `67.0`, LSTM `0.5262`, XGBoost `0.1637`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5244`, XGBoost `0.1623`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `53.0`, recent_utility `1`
- seconds `66.5`, LSTM `0.5260`, XGBoost `0.1645`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5272`, XGBoost `0.1662`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `7.0`, recent_utility `0`
