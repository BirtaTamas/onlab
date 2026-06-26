# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.734009 | 0.745132 | -0.011123 | 82 | 148 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.734009 | 0.745132 | -0.011123 | 82 | 148 | 1.000000 | 1.000000 |
| strong utility action | 168 | 0.730 | 0.666172 | 0.685400 | -0.019228 | 69 | 99 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 168 | 0.730 | 0.666172 | 0.685400 | -0.019228 | 69 | 99 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.983369 | 0.996294 | -0.012925 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.734009 | 0.745132 | -0.011123 | 82 | 148 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `90.0s`, rows `168`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `72.5`, LSTM `0.6874`, XGBoost `0.8114`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5371`, XGBoost `0.6506`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5453`, XGBoost `0.6573`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5378`, XGBoost `0.6457`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.5435`, XGBoost `0.6506`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5452`, XGBoost `0.6511`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.6752`, XGBoost `0.7810`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5449`, XGBoost `0.6506`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5401`, XGBoost `0.6457`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5407`, XGBoost `0.6457`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
