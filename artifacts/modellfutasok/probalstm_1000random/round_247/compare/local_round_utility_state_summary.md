# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `26`
- rows: `192`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.785562 | 0.800653 | -0.015090 | 58 | 134 | 0.963542 | 1.000000 |
| active/recent utility | 192 | 1.000 | 0.785562 | 0.800653 | -0.015090 | 58 | 134 | 0.963542 | 1.000000 |
| strong utility action | 135 | 0.703 | 0.748999 | 0.760472 | -0.011472 | 58 | 77 | 0.948148 | 1.000000 |
| utility damage | 21 | 0.109 | 0.533006 | 0.574356 | -0.041350 | 5 | 16 | 0.666667 | 1.000000 |
| active smoke/inferno | 135 | 0.703 | 0.748999 | 0.760472 | -0.011472 | 58 | 77 | 0.948148 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 192 | 1.000 | 0.785562 | 0.800653 | -0.015090 | 58 | 134 | 0.963542 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `73.5s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.6052`, XGBoost `0.7268`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `58.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4792`, XGBoost `0.5708`, closer `xgboost`, smoke `3`, inferno `5`, utility_damage `1.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4835`, XGBoost `0.5692`, closer `xgboost`, smoke `2`, inferno `4`, utility_damage `1.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4879`, XGBoost `0.5701`, closer `xgboost`, smoke `4`, inferno `5`, utility_damage `1.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6460`, XGBoost `0.7276`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `58.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.5000`, XGBoost `0.5737`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6566`, XGBoost `0.7279`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `58.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5050`, XGBoost `0.5750`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.8177`, XGBoost `0.7505`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.8180`, XGBoost `0.7512`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
