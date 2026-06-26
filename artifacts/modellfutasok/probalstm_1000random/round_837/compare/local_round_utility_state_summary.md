# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `20`
- rows: `240`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 240 | 1.000 | 0.096696 | 0.175509 | -0.078814 | 240 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 240 | 1.000 | 0.096696 | 0.175509 | -0.078814 | 240 | 0 | 1.000000 | 1.000000 |
| strong utility action | 146 | 0.608 | 0.137187 | 0.237606 | -0.100418 | 146 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 136 | 0.567 | 0.133228 | 0.234994 | -0.101766 | 136 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.042 | 0.191034 | 0.273128 | -0.082094 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 240 | 1.000 | 0.096696 | 0.175509 | -0.078814 | 240 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `32.5s`, rows `51`
- `44.0s` - `86.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.0`, LSTM `0.2311`, XGBoost `0.4844`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0535`, XGBoost `0.2814`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0568`, XGBoost `0.2814`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0587`, XGBoost `0.2814`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0327`, XGBoost `0.2489`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.2715`, XGBoost `0.4853`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0365`, XGBoost `0.2489`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0710`, XGBoost `0.2794`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.0726`, XGBoost `0.2700`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0861`, XGBoost `0.2790`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
