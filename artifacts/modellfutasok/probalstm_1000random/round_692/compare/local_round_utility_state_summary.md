# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-lynn-vision-bo3-0ZNMTlQ0ZfadRgwA0Ax5fN/3dmax-vs-lynn-vision-m2-anubis.csv`
- round_num: `14`
- rows: `204`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.614753 | 0.697128 | -0.082376 | 30 | 174 | 0.803922 | 0.833333 |
| active/recent utility | 204 | 1.000 | 0.614753 | 0.697128 | -0.082376 | 30 | 174 | 0.803922 | 0.833333 |
| strong utility action | 71 | 0.348 | 0.625987 | 0.712290 | -0.086303 | 12 | 59 | 0.676056 | 0.746479 |
| utility damage | 21 | 0.103 | 0.573164 | 0.676069 | -0.102905 | 0 | 21 | 0.523810 | 0.666667 |
| active smoke/inferno | 61 | 0.299 | 0.666103 | 0.747635 | -0.081532 | 12 | 49 | 0.786885 | 0.819672 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 204 | 1.000 | 0.614753 | 0.697128 | -0.082376 | 30 | 174 | 0.803922 | 0.833333 |

## Active Smoke/Inferno Intervals

- `8.0s` - `38.0s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.5`, LSTM `0.3490`, XGBoost `0.5725`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6551`, XGBoost `0.8369`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6577`, XGBoost `0.8369`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3944`, XGBoost `0.5728`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6612`, XGBoost `0.8369`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4027`, XGBoost `0.5751`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.4001`, XGBoost `0.5720`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4017`, XGBoost `0.5729`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6694`, XGBoost `0.8366`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4056`, XGBoost `0.5723`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
