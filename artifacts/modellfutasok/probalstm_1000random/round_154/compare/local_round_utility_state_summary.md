# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `15`
- rows: `123`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.919861 | 0.970024 | -0.050164 | 0 | 123 | 1.000000 | 1.000000 |
| active/recent utility | 123 | 1.000 | 0.919861 | 0.970024 | -0.050164 | 0 | 123 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.870 | 0.916663 | 0.969792 | -0.053128 | 0 | 107 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.171 | 0.895480 | 0.961503 | -0.066023 | 0 | 21 | 1.000000 | 1.000000 |
| active smoke/inferno | 107 | 0.870 | 0.916663 | 0.969792 | -0.053128 | 0 | 107 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 123 | 1.000 | 0.919861 | 0.970024 | -0.050164 | 0 | 123 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `59.0s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.8245`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8286`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.8353`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.8466`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.8552`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.8792`, XGBoost `0.9692`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.8692`, XGBoost `0.9553`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.8838`, XGBoost `0.9693`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.8849`, XGBoost `0.9692`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.8872`, XGBoost `0.9692`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
