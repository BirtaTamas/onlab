# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `14`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.892591 | 0.976034 | -0.083443 | 0 | 142 | 1.000000 | 1.000000 |
| active/recent utility | 142 | 1.000 | 0.892591 | 0.976034 | -0.083443 | 0 | 142 | 1.000000 | 1.000000 |
| strong utility action | 98 | 0.690 | 0.891084 | 0.977798 | -0.086715 | 0 | 98 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 98 | 0.690 | 0.891084 | 0.977798 | -0.086715 | 0 | 98 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.892591 | 0.976034 | -0.083443 | 0 | 142 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `54.0s`, rows `87`
- `61.0s` - `66.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.5`, LSTM `0.8451`, XGBoost `0.9788`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8456`, XGBoost `0.9788`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.8511`, XGBoost `0.9789`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8558`, XGBoost `0.9788`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.8585`, XGBoost `0.9789`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.8590`, XGBoost `0.9784`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.8601`, XGBoost `0.9789`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.8600`, XGBoost `0.9788`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8600`, XGBoost `0.9785`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.8621`, XGBoost `0.9784`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
