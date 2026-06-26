# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `3`
- rows: `116`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.932626 | 0.960972 | -0.028347 | 0 | 116 | 1.000000 | 1.000000 |
| active/recent utility | 116 | 1.000 | 0.932626 | 0.960972 | -0.028347 | 0 | 116 | 1.000000 | 1.000000 |
| strong utility action | 98 | 0.845 | 0.943707 | 0.964364 | -0.020657 | 0 | 98 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.086 | 0.960770 | 0.980654 | -0.019884 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 98 | 0.845 | 0.943707 | 0.964364 | -0.020657 | 0 | 98 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 116 | 1.000 | 0.932626 | 0.960972 | -0.028347 | 0 | 116 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `57.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.9039`, XGBoost `0.9497`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.9043`, XGBoost `0.9497`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9133`, XGBoost `0.9571`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9137`, XGBoost `0.9571`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.9145`, XGBoost `0.9556`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.9180`, XGBoost `0.9571`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.9128`, XGBoost `0.9497`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.9208`, XGBoost `0.9573`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.9143`, XGBoost `0.9494`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.9146`, XGBoost `0.9497`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
