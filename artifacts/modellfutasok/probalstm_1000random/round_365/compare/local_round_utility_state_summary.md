# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `16`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.551404 | 0.582591 | -0.031186 | 70 | 128 | 0.732323 | 0.898990 |
| active/recent utility | 198 | 1.000 | 0.551404 | 0.582591 | -0.031186 | 70 | 128 | 0.732323 | 0.898990 |
| strong utility action | 135 | 0.682 | 0.517240 | 0.549599 | -0.032359 | 58 | 77 | 0.659259 | 0.851852 |
| utility damage | 45 | 0.227 | 0.588541 | 0.632285 | -0.043744 | 20 | 25 | 0.888889 | 1.000000 |
| active smoke/inferno | 125 | 0.631 | 0.503904 | 0.525443 | -0.021539 | 58 | 67 | 0.632000 | 0.840000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.551404 | 0.582591 | -0.031186 | 70 | 128 | 0.732323 | 0.898990 |

## Active Smoke/Inferno Intervals

- `6.0s` - `62.5s`, rows `114`
- `64.0s` - `69.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.3501`, XGBoost `0.5964`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.5230`, XGBoost `0.7659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `34.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.3548`, XGBoost `0.5964`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.5392`, XGBoost `0.7659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `34.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.5458`, XGBoost `0.7642`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `34.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.5507`, XGBoost `0.7659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `34.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3856`, XGBoost `0.5964`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3935`, XGBoost `0.5964`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.5900`, XGBoost `0.7766`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `34.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.4182`, XGBoost `0.5964`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
