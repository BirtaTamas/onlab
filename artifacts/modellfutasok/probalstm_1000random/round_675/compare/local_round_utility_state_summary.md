# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `15`
- rows: `248`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 248 | 1.000 | 0.270061 | 0.258713 | 0.011348 | 135 | 113 | 0.564516 | 0.657258 |
| active/recent utility | 248 | 1.000 | 0.270061 | 0.258713 | 0.011348 | 135 | 113 | 0.564516 | 0.657258 |
| strong utility action | 155 | 0.625 | 0.370081 | 0.348698 | 0.021383 | 57 | 98 | 0.374194 | 0.496774 |
| utility damage | 21 | 0.085 | 0.044826 | 0.045379 | -0.000552 | 11 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.577 | 0.359086 | 0.341265 | 0.017821 | 55 | 88 | 0.391608 | 0.524476 |
| recent utility last 5s | 10 | 0.040 | 0.598047 | 0.520558 | 0.077489 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 248 | 1.000 | 0.270061 | 0.258713 | 0.011348 | 135 | 113 | 0.564516 | 0.657258 |

## Active Smoke/Inferno Intervals

- `8.5s` - `51.5s`, rows `87`
- `88.5s` - `110.5s`, rows `45`
- `117.5s` - `122.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.5311`, XGBoost `0.3935`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5328`, XGBoost `0.3962`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5251`, XGBoost `0.3908`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5240`, XGBoost `0.3899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5240`, XGBoost `0.3901`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5264`, XGBoost `0.3932`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5304`, XGBoost `0.3983`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5248`, XGBoost `0.3932`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5241`, XGBoost `0.3932`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5239`, XGBoost `0.3942`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
