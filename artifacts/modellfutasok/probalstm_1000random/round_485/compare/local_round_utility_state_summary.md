# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `13`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.438423 | 0.542181 | -0.103757 | 151 | 2 | 0.248366 | 0.189542 |
| active/recent utility | 153 | 1.000 | 0.438423 | 0.542181 | -0.103757 | 151 | 2 | 0.248366 | 0.189542 |
| strong utility action | 49 | 0.320 | 0.550625 | 0.728373 | -0.177748 | 49 | 0 | 0.142857 | 0.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 49 | 0.320 | 0.550625 | 0.728373 | -0.177748 | 49 | 0 | 0.142857 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.438423 | 0.542181 | -0.103757 | 151 | 2 | 0.248366 | 0.189542 |

## Active Smoke/Inferno Intervals

- `25.0s` - `49.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.4705`, XGBoost `0.7182`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.4803`, XGBoost `0.7196`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.4844`, XGBoost `0.7193`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4910`, XGBoost `0.7197`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.4934`, XGBoost `0.7197`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.4935`, XGBoost `0.7196`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4966`, XGBoost `0.7215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5053`, XGBoost `0.7156`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5044`, XGBoost `0.7144`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5117`, XGBoost `0.7215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
