# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `8`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.668069 | 0.603247 | 0.064821 | 172 | 30 | 0.980198 | 0.965347 |
| active/recent utility | 202 | 1.000 | 0.668069 | 0.603247 | 0.064821 | 172 | 30 | 0.980198 | 0.965347 |
| strong utility action | 164 | 0.812 | 0.645842 | 0.577074 | 0.068768 | 148 | 16 | 0.975610 | 0.957317 |
| utility damage | 20 | 0.099 | 0.605713 | 0.559780 | 0.045932 | 19 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 164 | 0.812 | 0.645842 | 0.577074 | 0.068768 | 148 | 16 | 0.975610 | 0.957317 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 202 | 1.000 | 0.668069 | 0.603247 | 0.064821 | 172 | 30 | 0.980198 | 0.965347 |

## Active Smoke/Inferno Intervals

- `8.0s` - `41.0s`, rows `67`
- `45.0s` - `93.0s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.5193`, XGBoost `0.3513`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5140`, XGBoost `0.3513`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.5058`, XGBoost `0.3509`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.4971`, XGBoost `0.3509`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4907`, XGBoost `0.3497`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.4848`, XGBoost `0.3497`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6964`, XGBoost `0.5617`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4844`, XGBoost `0.3509`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6904`, XGBoost `0.5590`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6926`, XGBoost `0.5617`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
