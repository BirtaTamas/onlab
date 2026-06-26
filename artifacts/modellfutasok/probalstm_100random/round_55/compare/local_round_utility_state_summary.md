# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `8`
- rows: `280`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 280 | 1.000 | 0.039737 | 0.097988 | -0.058251 | 280 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 280 | 1.000 | 0.039737 | 0.097988 | -0.058251 | 280 | 0 | 1.000000 | 1.000000 |
| strong utility action | 216 | 0.771 | 0.043345 | 0.112492 | -0.069146 | 216 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.036 | 0.061280 | 0.187262 | -0.125982 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 206 | 0.736 | 0.038274 | 0.104643 | -0.066369 | 206 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.036 | 0.147823 | 0.274179 | -0.126355 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 280 | 1.000 | 0.039737 | 0.097988 | -0.058251 | 280 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `113.0s`, rows `206`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.0676`, XGBoost `0.2546`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `148.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0500`, XGBoost `0.2325`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `139.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0651`, XGBoost `0.2472`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `130.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1069`, XGBoost `0.2797`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `15.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0950`, XGBoost `0.2627`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `50.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1124`, XGBoost `0.2786`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `15.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0928`, XGBoost `0.2570`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `49.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1180`, XGBoost `0.2797`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `15.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.1185`, XGBoost `0.2797`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `15.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1206`, XGBoost `0.2786`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `15.0`, recent_utility `0`
