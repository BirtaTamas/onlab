# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `7`
- rows: `203`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.285330 | 0.333658 | -0.048327 | 173 | 30 | 0.886700 | 0.911330 |
| active/recent utility | 203 | 1.000 | 0.285330 | 0.333658 | -0.048327 | 173 | 30 | 0.886700 | 0.911330 |
| strong utility action | 168 | 0.828 | 0.273959 | 0.328919 | -0.054959 | 144 | 24 | 0.863095 | 0.892857 |
| utility damage | 15 | 0.074 | 0.536807 | 0.521856 | 0.014951 | 2 | 13 | 0.133333 | 0.133333 |
| active smoke/inferno | 168 | 0.828 | 0.273959 | 0.328919 | -0.054959 | 144 | 24 | 0.863095 | 0.892857 |
| recent utility last 5s | 10 | 0.049 | 0.004713 | 0.034482 | -0.029769 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 203 | 1.000 | 0.285330 | 0.333658 | -0.048327 | 173 | 30 | 0.886700 | 0.911330 |

## Active Smoke/Inferno Intervals

- `6.5s` - `39.0s`, rows `66`
- `45.5s` - `96.0s`, rows `102`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.2009`, XGBoost `0.5242`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.2081`, XGBoost `0.5259`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `72.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.2307`, XGBoost `0.4971`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0123`, XGBoost `0.2208`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0139`, XGBoost `0.2105`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0152`, XGBoost `0.2080`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.0178`, XGBoost `0.2080`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5322`, XGBoost `0.6983`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `38.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3030`, XGBoost `0.4662`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `25.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.0364`, XGBoost `0.1975`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `47.0`, recent_utility `0`
