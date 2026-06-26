# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `7`
- rows: `183`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.781205 | 0.783328 | -0.002123 | 77 | 106 | 1.000000 | 1.000000 |
| active/recent utility | 183 | 1.000 | 0.781205 | 0.783328 | -0.002123 | 77 | 106 | 1.000000 | 1.000000 |
| strong utility action | 158 | 0.863 | 0.772245 | 0.775835 | -0.003590 | 65 | 93 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.109 | 0.861021 | 0.868655 | -0.007634 | 3 | 17 | 1.000000 | 1.000000 |
| active smoke/inferno | 147 | 0.803 | 0.758773 | 0.761499 | -0.002726 | 65 | 82 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.781205 | 0.783328 | -0.002123 | 77 | 106 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `34.0s`, rows `55`
- `35.0s` - `80.5s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.7073`, XGBoost `0.7890`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.8630`, XGBoost `0.7850`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.8617`, XGBoost `0.7850`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.8640`, XGBoost `0.7893`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.7148`, XGBoost `0.7890`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.8570`, XGBoost `0.7832`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.8583`, XGBoost `0.7850`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.8614`, XGBoost `0.7887`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.8534`, XGBoost `0.7834`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.8527`, XGBoost `0.7853`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
