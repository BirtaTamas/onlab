# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `8`
- rows: `225`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.740613 | 0.784735 | -0.044122 | 10 | 215 | 0.911111 | 1.000000 |
| active/recent utility | 225 | 1.000 | 0.740613 | 0.784735 | -0.044122 | 10 | 215 | 0.911111 | 1.000000 |
| strong utility action | 191 | 0.849 | 0.755565 | 0.805089 | -0.049524 | 10 | 181 | 0.958115 | 1.000000 |
| utility damage | 10 | 0.044 | 0.951973 | 0.990930 | -0.038958 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 191 | 0.849 | 0.755565 | 0.805089 | -0.049524 | 10 | 181 | 0.958115 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 225 | 1.000 | 0.740613 | 0.784735 | -0.044122 | 10 | 215 | 0.911111 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `82.0s`, rows `145`
- `89.5s` - `112.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.0`, LSTM `0.5642`, XGBoost `0.7351`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5669`, XGBoost `0.7377`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5763`, XGBoost `0.7372`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5776`, XGBoost `0.7372`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5887`, XGBoost `0.7446`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5918`, XGBoost `0.7430`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5952`, XGBoost `0.7446`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5953`, XGBoost `0.7446`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5886`, XGBoost `0.7372`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5936`, XGBoost `0.7397`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
