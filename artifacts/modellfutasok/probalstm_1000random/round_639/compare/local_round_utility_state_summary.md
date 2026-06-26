# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `3`
- rows: `114`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 114 | 1.000 | 0.416058 | 0.576060 | -0.160002 | 108 | 6 | 0.789474 | 0.421053 |
| active/recent utility | 114 | 1.000 | 0.416058 | 0.576060 | -0.160002 | 108 | 6 | 0.789474 | 0.421053 |
| strong utility action | 90 | 0.789 | 0.460889 | 0.583914 | -0.123025 | 88 | 2 | 0.733333 | 0.411111 |
| utility damage | 11 | 0.096 | 0.551465 | 0.666977 | -0.115512 | 11 | 0 | 0.090909 | 0.000000 |
| active smoke/inferno | 82 | 0.719 | 0.460774 | 0.593567 | -0.132793 | 80 | 2 | 0.707317 | 0.353659 |
| recent utility last 5s | 10 | 0.088 | 0.458107 | 0.481024 | -0.022917 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 114 | 1.000 | 0.416058 | 0.576060 | -0.160002 | 108 | 6 | 0.789474 | 0.421053 |

## Active Smoke/Inferno Intervals

- `6.0s` - `46.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `46.5`, LSTM `0.3414`, XGBoost `0.6867`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.3760`, XGBoost `0.6867`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.3868`, XGBoost `0.6549`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.3869`, XGBoost `0.6506`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.4257`, XGBoost `0.6867`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3943`, XGBoost `0.6462`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.4103`, XGBoost `0.6549`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.4159`, XGBoost `0.6549`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.4216`, XGBoost `0.6549`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4235`, XGBoost `0.6549`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
