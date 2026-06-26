# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `15`
- rows: `186`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.944914 | 0.979200 | -0.034286 | 0 | 186 | 1.000000 | 1.000000 |
| active/recent utility | 186 | 1.000 | 0.944914 | 0.979200 | -0.034286 | 0 | 186 | 1.000000 | 1.000000 |
| strong utility action | 95 | 0.511 | 0.936709 | 0.977175 | -0.040466 | 0 | 95 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.108 | 0.930767 | 0.974533 | -0.043765 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 94 | 0.505 | 0.936674 | 0.977175 | -0.040501 | 0 | 94 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 186 | 1.000 | 0.944914 | 0.979200 | -0.034286 | 0 | 186 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `33.0s`, rows `50`
- `59.0s` - `80.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.9013`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `16.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9024`, XGBoost `0.9722`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.9050`, XGBoost `0.9722`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.9103`, XGBoost `0.9722`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9111`, XGBoost `0.9721`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9159`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9174`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.9175`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.9251`, XGBoost `0.9782`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.9298`, XGBoost `0.9784`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
