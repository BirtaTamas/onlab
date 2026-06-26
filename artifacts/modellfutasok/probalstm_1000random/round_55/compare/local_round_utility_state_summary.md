# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `14`
- rows: `147`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 147 | 1.000 | 0.927747 | 0.983762 | -0.056015 | 0 | 147 | 1.000000 | 1.000000 |
| active/recent utility | 147 | 1.000 | 0.927747 | 0.983762 | -0.056015 | 0 | 147 | 1.000000 | 1.000000 |
| strong utility action | 79 | 0.537 | 0.918514 | 0.981457 | -0.062943 | 0 | 79 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 79 | 0.537 | 0.918514 | 0.981457 | -0.062943 | 0 | 79 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.068 | 0.973530 | 0.995114 | -0.021584 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 147 | 1.000 | 0.927747 | 0.983762 | -0.056015 | 0 | 147 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `19.0s`, rows `22`
- `25.5s` - `53.5s`, rows `57`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.8291`, XGBoost `0.9741`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.8302`, XGBoost `0.9736`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.8303`, XGBoost `0.9736`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.8305`, XGBoost `0.9736`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.8311`, XGBoost `0.9741`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.8341`, XGBoost `0.9736`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.8372`, XGBoost `0.9741`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.8392`, XGBoost `0.9736`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.8414`, XGBoost `0.9745`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.8465`, XGBoost `0.9741`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
