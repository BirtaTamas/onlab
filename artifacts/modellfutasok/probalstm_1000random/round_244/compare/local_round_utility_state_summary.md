# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `20`
- rows: `236`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 236 | 1.000 | 0.637213 | 0.717263 | -0.080051 | 37 | 199 | 0.949153 | 1.000000 |
| active/recent utility | 236 | 1.000 | 0.637213 | 0.717263 | -0.080051 | 37 | 199 | 0.949153 | 1.000000 |
| strong utility action | 209 | 0.886 | 0.629846 | 0.716604 | -0.086759 | 24 | 185 | 0.942584 | 1.000000 |
| utility damage | 11 | 0.047 | 0.666492 | 0.641792 | 0.024701 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 209 | 0.886 | 0.629846 | 0.716604 | -0.086759 | 24 | 185 | 0.942584 | 1.000000 |
| recent utility last 5s | 10 | 0.042 | 0.786467 | 0.895412 | -0.108944 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 236 | 1.000 | 0.637213 | 0.717263 | -0.080051 | 37 | 199 | 0.949153 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `66.0s`, rows `119`
- `72.0s` - `77.0s`, rows `11`
- `78.0s` - `117.0s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `101.5`, LSTM `0.4127`, XGBoost `0.7893`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.4126`, XGBoost `0.7885`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.4145`, XGBoost `0.7885`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.4201`, XGBoost `0.7893`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.4654`, XGBoost `0.7893`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.3381`, XGBoost `0.6518`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.3438`, XGBoost `0.6518`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.4786`, XGBoost `0.7852`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.4842`, XGBoost `0.7876`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.4600`, XGBoost `0.7502`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
