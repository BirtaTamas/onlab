# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `13`
- rows: `110`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 110 | 1.000 | 0.621349 | 0.766595 | -0.145246 | 0 | 110 | 0.945455 | 1.000000 |
| active/recent utility | 110 | 1.000 | 0.621349 | 0.766595 | -0.145246 | 0 | 110 | 0.945455 | 1.000000 |
| strong utility action | 44 | 0.400 | 0.589977 | 0.777710 | -0.187734 | 0 | 44 | 0.863636 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.400 | 0.589977 | 0.777710 | -0.187734 | 0 | 44 | 0.863636 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 110 | 1.000 | 0.621349 | 0.766595 | -0.145246 | 0 | 110 | 0.945455 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `32.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.4475`, XGBoost `0.7477`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4582`, XGBoost `0.7459`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6047`, XGBoost `0.8918`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6054`, XGBoost `0.8918`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.4651`, XGBoost `0.7485`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6110`, XGBoost `0.8898`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6178`, XGBoost `0.8896`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6357`, XGBoost `0.8899`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.6390`, XGBoost `0.8895`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6423`, XGBoost `0.8906`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
