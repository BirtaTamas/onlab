# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `24`
- rows: `243`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 243 | 1.000 | 0.148039 | 0.241845 | -0.093806 | 243 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 243 | 1.000 | 0.148039 | 0.241845 | -0.093806 | 243 | 0 | 1.000000 | 1.000000 |
| strong utility action | 209 | 0.860 | 0.154780 | 0.251393 | -0.096613 | 209 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.041 | 0.219043 | 0.321269 | -0.102226 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 209 | 0.860 | 0.154780 | 0.251393 | -0.096613 | 209 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.041 | 0.202151 | 0.291877 | -0.089726 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 243 | 1.000 | 0.148039 | 0.241845 | -0.093806 | 243 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `113.5s`, rows `209`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `90.0`, LSTM `0.0367`, XGBoost `0.2776`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0266`, XGBoost `0.2612`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.0309`, XGBoost `0.2510`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.0503`, XGBoost `0.2639`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1075`, XGBoost `0.2897`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1080`, XGBoost `0.2897`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0929`, XGBoost `0.2709`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1167`, XGBoost `0.2897`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.1177`, XGBoost `0.2897`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1214`, XGBoost `0.2897`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
