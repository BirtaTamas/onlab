# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `1`
- rows: `102`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 102 | 1.000 | 0.680655 | 0.733450 | -0.052795 | 26 | 76 | 0.980392 | 1.000000 |
| active/recent utility | 102 | 1.000 | 0.680655 | 0.733450 | -0.052795 | 26 | 76 | 0.980392 | 1.000000 |
| strong utility action | 44 | 0.431 | 0.697105 | 0.744918 | -0.047813 | 6 | 38 | 0.954545 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.431 | 0.697105 | 0.744918 | -0.047813 | 6 | 38 | 0.954545 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 102 | 1.000 | 0.680655 | 0.733450 | -0.052795 | 26 | 76 | 0.980392 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `32.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.5737`, XGBoost `0.7240`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5768`, XGBoost `0.7248`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5823`, XGBoost `0.7230`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6058`, XGBoost `0.7305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5865`, XGBoost `0.7082`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6043`, XGBoost `0.7260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6119`, XGBoost `0.7248`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6099`, XGBoost `0.7215`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6224`, XGBoost `0.7305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.7414`, XGBoost `0.8439`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
