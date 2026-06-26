# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `28`
- rows: `208`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.445249 | 0.558305 | -0.113056 | 50 | 158 | 0.581731 | 0.649038 |
| active/recent utility | 208 | 1.000 | 0.445249 | 0.558305 | -0.113056 | 50 | 158 | 0.581731 | 0.649038 |
| strong utility action | 163 | 0.784 | 0.404407 | 0.535000 | -0.130594 | 29 | 134 | 0.558282 | 0.625767 |
| utility damage | 27 | 0.130 | 0.521146 | 0.523057 | -0.001912 | 14 | 13 | 0.629630 | 0.629630 |
| active smoke/inferno | 156 | 0.750 | 0.398181 | 0.535538 | -0.137356 | 23 | 133 | 0.538462 | 0.608974 |
| recent utility last 5s | 10 | 0.048 | 0.532573 | 0.522591 | 0.009982 | 6 | 4 | 1.000000 | 1.000000 |
| flash effect present | 208 | 1.000 | 0.445249 | 0.558305 | -0.113056 | 50 | 158 | 0.581731 | 0.649038 |

## Active Smoke/Inferno Intervals

- `6.5s` - `55.0s`, rows `98`
- `61.5s` - `90.0s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `85.0`, LSTM `0.0844`, XGBoost `0.4568`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1034`, XGBoost `0.4710`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.3969`, XGBoost `0.7375`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.4135`, XGBoost `0.7396`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.4293`, XGBoost `0.7378`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.0870`, XGBoost `0.3665`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.0307`, XGBoost `0.2919`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0381`, XGBoost `0.2981`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.0330`, XGBoost `0.2919`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0337`, XGBoost `0.2919`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
