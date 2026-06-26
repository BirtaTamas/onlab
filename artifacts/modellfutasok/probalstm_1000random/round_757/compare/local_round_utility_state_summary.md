# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `4`
- rows: `152`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.194992 | 0.250219 | -0.055227 | 127 | 25 | 0.907895 | 0.743421 |
| active/recent utility | 152 | 1.000 | 0.194992 | 0.250219 | -0.055227 | 127 | 25 | 0.907895 | 0.743421 |
| strong utility action | 105 | 0.691 | 0.207414 | 0.252479 | -0.045065 | 82 | 23 | 0.866667 | 0.742857 |
| utility damage | 20 | 0.132 | 0.327990 | 0.361142 | -0.033152 | 10 | 10 | 0.600000 | 0.500000 |
| active smoke/inferno | 105 | 0.691 | 0.207414 | 0.252479 | -0.045065 | 82 | 23 | 0.866667 | 0.742857 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 152 | 1.000 | 0.194992 | 0.250219 | -0.055227 | 127 | 25 | 0.907895 | 0.743421 |

## Active Smoke/Inferno Intervals

- `6.0s` - `36.0s`, rows `61`
- `51.0s` - `72.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.3600`, XGBoost `0.5372`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3918`, XGBoost `0.5392`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1495`, XGBoost `0.2792`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5125`, XGBoost `0.6398`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `64.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5192`, XGBoost `0.6398`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `75.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1594`, XGBoost `0.2786`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0336`, XGBoost `0.1516`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0351`, XGBoost `0.1516`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.5262`, XGBoost `0.6398`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `75.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0403`, XGBoost `0.1516`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
