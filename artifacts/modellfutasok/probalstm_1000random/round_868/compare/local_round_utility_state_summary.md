# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`
- rows: `145`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 145 | 1.000 | 0.586491 | 0.555476 | 0.031015 | 102 | 43 | 0.910345 | 0.441379 |
| active/recent utility | 145 | 1.000 | 0.586491 | 0.555476 | 0.031015 | 102 | 43 | 0.910345 | 0.441379 |
| strong utility action | 139 | 0.959 | 0.581791 | 0.548747 | 0.033044 | 97 | 42 | 0.913669 | 0.417266 |
| utility damage | 16 | 0.110 | 0.505866 | 0.486215 | 0.019651 | 11 | 5 | 0.687500 | 0.437500 |
| active smoke/inferno | 131 | 0.903 | 0.573196 | 0.538799 | 0.034398 | 91 | 40 | 0.908397 | 0.381679 |
| recent utility last 5s | 20 | 0.138 | 0.646299 | 0.589771 | 0.056527 | 18 | 2 | 1.000000 | 0.500000 |
| flash effect present | 145 | 1.000 | 0.586491 | 0.555476 | 0.031015 | 102 | 43 | 0.910345 | 0.441379 |

## Active Smoke/Inferno Intervals

- `6.5s` - `71.5s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `71.5`, LSTM `0.4815`, XGBoost `0.6896`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.4865`, XGBoost `0.6890`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5036`, XGBoost `0.6832`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.6184`, XGBoost `0.4438`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.8236`, XGBoost `0.6555`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6044`, XGBoost `0.4380`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5251`, XGBoost `0.6844`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6003`, XGBoost `0.4426`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5293`, XGBoost `0.6864`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7856`, XGBoost `0.6295`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
