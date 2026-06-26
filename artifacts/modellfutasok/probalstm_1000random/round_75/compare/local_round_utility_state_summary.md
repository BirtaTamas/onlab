# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `5`
- rows: `181`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.860112 | 0.851024 | 0.009088 | 85 | 96 | 1.000000 | 1.000000 |
| active/recent utility | 181 | 1.000 | 0.860112 | 0.851024 | 0.009088 | 85 | 96 | 1.000000 | 1.000000 |
| strong utility action | 156 | 0.862 | 0.852519 | 0.843242 | 0.009277 | 77 | 79 | 1.000000 | 1.000000 |
| utility damage | 38 | 0.210 | 0.930655 | 0.933168 | -0.002513 | 8 | 30 | 1.000000 | 1.000000 |
| active smoke/inferno | 146 | 0.807 | 0.860067 | 0.851291 | 0.008776 | 67 | 79 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.055 | 0.742316 | 0.725722 | 0.016594 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 181 | 1.000 | 0.860112 | 0.851024 | 0.009088 | 85 | 96 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `76.0s`, rows `135`
- `83.5s` - `88.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.7407`, XGBoost `0.5810`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `36.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6932`, XGBoost `0.5752`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `36.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.8838`, XGBoost `0.7681`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `62.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.8653`, XGBoost `0.7543`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6806`, XGBoost `0.5724`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `36.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.8726`, XGBoost `0.7657`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `62.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6733`, XGBoost `0.5752`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `36.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8432`, XGBoost `0.7453`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.8563`, XGBoost `0.7612`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7539`, XGBoost `0.6624`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `36.0`, recent_utility `0`
