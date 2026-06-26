# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `10`
- rows: `172`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 172 | 1.000 | 0.670026 | 0.600455 | 0.069572 | 146 | 26 | 0.912791 | 0.901163 |
| active/recent utility | 172 | 1.000 | 0.670026 | 0.600455 | 0.069572 | 146 | 26 | 0.912791 | 0.901163 |
| strong utility action | 128 | 0.744 | 0.657780 | 0.581801 | 0.075979 | 111 | 17 | 0.882812 | 0.867188 |
| utility damage | 20 | 0.116 | 0.585680 | 0.531850 | 0.053831 | 15 | 5 | 0.700000 | 0.900000 |
| active smoke/inferno | 118 | 0.686 | 0.656436 | 0.578485 | 0.077951 | 101 | 17 | 0.872881 | 0.855932 |
| recent utility last 5s | 20 | 0.116 | 0.603921 | 0.577532 | 0.026390 | 16 | 4 | 0.900000 | 1.000000 |
| flash effect present | 172 | 1.000 | 0.670026 | 0.600455 | 0.069572 | 146 | 26 | 0.912791 | 0.901163 |

## Active Smoke/Inferno Intervals

- `9.0s` - `34.5s`, rows `52`
- `46.0s` - `78.5s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.7039`, XGBoost `0.4698`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6986`, XGBoost `0.4698`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7034`, XGBoost `0.4797`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7043`, XGBoost `0.4837`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6891`, XGBoost `0.4698`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6897`, XGBoost `0.4849`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1553`, XGBoost `0.3565`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6783`, XGBoost `0.4805`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.7046`, XGBoost `0.5269`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.1925`, XGBoost `0.3580`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
