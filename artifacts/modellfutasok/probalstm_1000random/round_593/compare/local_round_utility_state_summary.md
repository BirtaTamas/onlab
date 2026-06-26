# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `14`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.522724 | 0.608436 | -0.085711 | 12 | 200 | 0.783019 | 0.900943 |
| active/recent utility | 212 | 1.000 | 0.522724 | 0.608436 | -0.085711 | 12 | 200 | 0.783019 | 0.900943 |
| strong utility action | 186 | 0.877 | 0.516053 | 0.604135 | -0.088082 | 12 | 174 | 0.763441 | 0.887097 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 186 | 0.877 | 0.516053 | 0.604135 | -0.088082 | 12 | 174 | 0.763441 | 0.887097 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.522724 | 0.608436 | -0.085711 | 12 | 200 | 0.783019 | 0.900943 |

## Active Smoke/Inferno Intervals

- `8.5s` - `41.5s`, rows `67`
- `43.0s` - `78.0s`, rows `71`
- `80.0s` - `103.5s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `97.5`, LSTM `0.3732`, XGBoost `0.6494`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.5250`, XGBoost `0.7954`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5446`, XGBoost `0.7954`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4792`, XGBoost `0.7262`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.3928`, XGBoost `0.6289`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.6812`, XGBoost `0.9096`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.4292`, XGBoost `0.6494`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.4180`, XGBoost `0.6379`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.5752`, XGBoost `0.7950`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5152`, XGBoost `0.7262`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
