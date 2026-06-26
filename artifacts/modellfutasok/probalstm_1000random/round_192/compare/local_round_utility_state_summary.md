# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `9`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.747665 | 0.825447 | -0.077781 | 12 | 218 | 0.943478 | 0.934783 |
| active/recent utility | 230 | 1.000 | 0.747665 | 0.825447 | -0.077781 | 12 | 218 | 0.943478 | 0.934783 |
| strong utility action | 182 | 0.791 | 0.735408 | 0.816826 | -0.081418 | 7 | 175 | 0.928571 | 0.917582 |
| utility damage | 10 | 0.043 | 0.850275 | 0.952913 | -0.102638 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 170 | 0.739 | 0.737170 | 0.821558 | -0.084388 | 4 | 166 | 0.923529 | 0.911765 |
| recent utility last 5s | 14 | 0.061 | 0.704958 | 0.751704 | -0.046747 | 3 | 11 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.747665 | 0.825447 | -0.077781 | 12 | 218 | 0.943478 | 0.934783 |

## Active Smoke/Inferno Intervals

- `6.5s` - `65.0s`, rows `118`
- `78.5s` - `104.0s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.0`, LSTM `0.5758`, XGBoost `0.8715`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.7666`, XGBoost `0.9510`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.7713`, XGBoost `0.9510`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.7890`, XGBoost `0.9525`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.7926`, XGBoost `0.9496`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7291`, XGBoost `0.8841`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7324`, XGBoost `0.8844`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7120`, XGBoost `0.8596`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7380`, XGBoost `0.8844`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.8026`, XGBoost `0.9486`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
