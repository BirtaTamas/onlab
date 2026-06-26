# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `8`
- rows: `140`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.263676 | 0.286847 | -0.023171 | 105 | 35 | 0.642857 | 0.635714 |
| active/recent utility | 140 | 1.000 | 0.263676 | 0.286847 | -0.023171 | 105 | 35 | 0.642857 | 0.635714 |
| strong utility action | 75 | 0.536 | 0.380714 | 0.422991 | -0.042277 | 52 | 23 | 0.493333 | 0.480000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 75 | 0.536 | 0.380714 | 0.422991 | -0.042277 | 52 | 23 | 0.493333 | 0.480000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 140 | 1.000 | 0.263676 | 0.286847 | -0.023171 | 105 | 35 | 0.642857 | 0.635714 |

## Active Smoke/Inferno Intervals

- `6.0s` - `43.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.0204`, XGBoost `0.2102`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0203`, XGBoost `0.1844`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0192`, XGBoost `0.1805`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0227`, XGBoost `0.1808`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0231`, XGBoost `0.1808`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0228`, XGBoost `0.1789`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0280`, XGBoost `0.1810`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0209`, XGBoost `0.1729`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0346`, XGBoost `0.1810`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0359`, XGBoost `0.1810`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
