# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `13`
- rows: `131`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 131 | 1.000 | 0.458567 | 0.587563 | -0.128996 | 3 | 128 | 0.603053 | 0.832061 |
| active/recent utility | 131 | 1.000 | 0.458567 | 0.587563 | -0.128996 | 3 | 128 | 0.603053 | 0.832061 |
| strong utility action | 75 | 0.573 | 0.416536 | 0.571177 | -0.154641 | 3 | 72 | 0.386667 | 0.706667 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 75 | 0.573 | 0.416536 | 0.571177 | -0.154641 | 3 | 72 | 0.386667 | 0.706667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 131 | 1.000 | 0.458567 | 0.587563 | -0.128996 | 3 | 128 | 0.603053 | 0.832061 |

## Active Smoke/Inferno Intervals

- `20.0s` - `57.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.2694`, XGBoost `0.6859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2780`, XGBoost `0.6810`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3204`, XGBoost `0.6975`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3056`, XGBoost `0.6810`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3215`, XGBoost `0.6936`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.3803`, XGBoost `0.7415`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3556`, XGBoost `0.6985`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3882`, XGBoost `0.7271`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3670`, XGBoost `0.6961`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3718`, XGBoost `0.6913`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
