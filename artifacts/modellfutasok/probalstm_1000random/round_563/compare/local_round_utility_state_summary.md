# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `25`
- rows: `216`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.425666 | 0.421581 | 0.004085 | 86 | 130 | 0.305556 | 0.277778 |
| active/recent utility | 216 | 1.000 | 0.425666 | 0.421581 | 0.004085 | 86 | 130 | 0.305556 | 0.277778 |
| strong utility action | 202 | 0.935 | 0.441209 | 0.434933 | 0.006275 | 76 | 126 | 0.272277 | 0.242574 |
| utility damage | 20 | 0.093 | 0.554171 | 0.531735 | 0.022436 | 3 | 17 | 0.000000 | 0.000000 |
| active smoke/inferno | 191 | 0.884 | 0.434267 | 0.429388 | 0.004880 | 76 | 115 | 0.287958 | 0.256545 |
| recent utility last 5s | 11 | 0.051 | 0.561740 | 0.531230 | 0.030510 | 0 | 11 | 0.000000 | 0.000000 |
| flash effect present | 216 | 1.000 | 0.425666 | 0.421581 | 0.004085 | 86 | 130 | 0.305556 | 0.277778 |

## Active Smoke/Inferno Intervals

- `6.5s` - `77.0s`, rows `142`
- `79.0s` - `103.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.5060`, XGBoost `0.3482`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.4934`, XGBoost `0.3435`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.4798`, XGBoost `0.3763`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5082`, XGBoost `0.4408`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `0.5`, LSTM `0.5832`, XGBoost `0.5229`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `101.5`, LSTM `0.0196`, XGBoost `0.0790`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.0204`, XGBoost `0.0792`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.4975`, XGBoost `0.4408`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.0227`, XGBoost `0.0792`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.0229`, XGBoost `0.0788`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
