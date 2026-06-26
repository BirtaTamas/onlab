# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `16`
- rows: `224`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.192933 | 0.255918 | -0.062985 | 182 | 42 | 0.941964 | 0.919643 |
| active/recent utility | 224 | 1.000 | 0.192933 | 0.255918 | -0.062985 | 182 | 42 | 0.941964 | 0.919643 |
| strong utility action | 204 | 0.911 | 0.169425 | 0.241292 | -0.071867 | 179 | 25 | 0.936275 | 0.911765 |
| utility damage | 10 | 0.045 | 0.285345 | 0.410255 | -0.124910 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 204 | 0.911 | 0.169425 | 0.241292 | -0.071867 | 179 | 25 | 0.936275 | 0.911765 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.192933 | 0.255918 | -0.062985 | 182 | 42 | 0.941964 | 0.919643 |

## Active Smoke/Inferno Intervals

- `10.0s` - `111.5s`, rows `204`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.1305`, XGBoost `0.3944`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1350`, XGBoost `0.3937`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.1364`, XGBoost `0.3937`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1398`, XGBoost `0.3964`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1284`, XGBoost `0.3839`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1471`, XGBoost `0.3993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1476`, XGBoost `0.3996`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1342`, XGBoost `0.3839`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1698`, XGBoost `0.4180`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1604`, XGBoost `0.4076`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
