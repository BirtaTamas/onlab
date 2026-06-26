# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `19`
- rows: `197`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.774358 | 0.792306 | -0.017948 | 59 | 138 | 1.000000 | 1.000000 |
| active/recent utility | 197 | 1.000 | 0.774358 | 0.792306 | -0.017948 | 59 | 138 | 1.000000 | 1.000000 |
| strong utility action | 133 | 0.675 | 0.740680 | 0.767451 | -0.026771 | 35 | 98 | 1.000000 | 1.000000 |
| utility damage | 29 | 0.147 | 0.725727 | 0.744409 | -0.018682 | 3 | 26 | 1.000000 | 1.000000 |
| active smoke/inferno | 132 | 0.670 | 0.740791 | 0.767738 | -0.026947 | 35 | 97 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.774358 | 0.792306 | -0.017948 | 59 | 138 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `39.5s`, rows `61`
- `44.0s` - `49.0s`, rows `11`
- `51.5s` - `81.0s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.5`, LSTM `0.7851`, XGBoost `0.9607`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.8021`, XGBoost `0.9542`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.8157`, XGBoost `0.9599`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.8101`, XGBoost `0.9542`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.8105`, XGBoost `0.9544`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.8225`, XGBoost `0.9591`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.8273`, XGBoost `0.9591`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.8202`, XGBoost `0.9518`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.8191`, XGBoost `0.9490`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.8352`, XGBoost `0.9515`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
