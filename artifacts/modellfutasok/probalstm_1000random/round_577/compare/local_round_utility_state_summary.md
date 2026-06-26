# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `2`
- rows: `221`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 221 | 1.000 | 0.077565 | 0.118618 | -0.041053 | 219 | 2 | 1.000000 | 1.000000 |
| active/recent utility | 221 | 1.000 | 0.077565 | 0.118618 | -0.041053 | 219 | 2 | 1.000000 | 1.000000 |
| strong utility action | 156 | 0.706 | 0.093496 | 0.144898 | -0.051402 | 156 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 146 | 0.661 | 0.082497 | 0.132196 | -0.049700 | 146 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.090 | 0.148253 | 0.233352 | -0.085100 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 221 | 1.000 | 0.077565 | 0.118618 | -0.041053 | 219 | 2 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `32.0s`, rows `45`
- `33.0s` - `83.0s`, rows `101`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.0768`, XGBoost `0.2663`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0918`, XGBoost `0.2663`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0919`, XGBoost `0.2658`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `47.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0996`, XGBoost `0.2656`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `47.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1065`, XGBoost `0.2663`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1146`, XGBoost `0.2663`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1172`, XGBoost `0.2663`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.1213`, XGBoost `0.2663`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1685`, XGBoost `0.3096`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1706`, XGBoost `0.3099`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
