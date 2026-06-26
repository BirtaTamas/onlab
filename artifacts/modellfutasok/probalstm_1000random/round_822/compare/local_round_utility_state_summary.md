# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `15`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.711816 | 0.668393 | 0.043423 | 139 | 26 | 1.000000 | 0.654545 |
| active/recent utility | 165 | 1.000 | 0.711816 | 0.668393 | 0.043423 | 139 | 26 | 1.000000 | 0.654545 |
| strong utility action | 153 | 0.927 | 0.723902 | 0.682221 | 0.041680 | 127 | 26 | 1.000000 | 0.705882 |
| utility damage | 10 | 0.061 | 0.838899 | 0.765609 | 0.073290 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 153 | 0.927 | 0.723902 | 0.682221 | 0.041680 | 127 | 26 | 1.000000 | 0.705882 |
| recent utility last 5s | 10 | 0.061 | 0.799840 | 0.788867 | 0.010973 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 165 | 1.000 | 0.711816 | 0.668393 | 0.043423 | 139 | 26 | 1.000000 | 0.654545 |

## Active Smoke/Inferno Intervals

- `6.0s` - `82.0s`, rows `153`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.5`, LSTM `0.6140`, XGBoost `0.4961`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6090`, XGBoost `0.4915`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.8257`, XGBoost `0.9413`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6113`, XGBoost `0.4968`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.8164`, XGBoost `0.9286`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6034`, XGBoost `0.4915`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6084`, XGBoost `0.4968`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.8334`, XGBoost `0.9431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.8312`, XGBoost `0.9408`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5732`, XGBoost `0.4647`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
