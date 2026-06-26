# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `5`
- rows: `223`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.470874 | 0.541033 | -0.070159 | 217 | 6 | 0.834081 | 0.327354 |
| active/recent utility | 223 | 1.000 | 0.470874 | 0.541033 | -0.070159 | 217 | 6 | 0.834081 | 0.327354 |
| strong utility action | 194 | 0.870 | 0.496422 | 0.560123 | -0.063701 | 188 | 6 | 0.809278 | 0.278351 |
| utility damage | 20 | 0.090 | 0.506607 | 0.577609 | -0.071002 | 17 | 3 | 0.500000 | 0.150000 |
| active smoke/inferno | 194 | 0.870 | 0.496422 | 0.560123 | -0.063701 | 188 | 6 | 0.809278 | 0.278351 |
| recent utility last 5s | 10 | 0.045 | 0.418972 | 0.503554 | -0.084582 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 223 | 1.000 | 0.470874 | 0.541033 | -0.070159 | 217 | 6 | 0.834081 | 0.327354 |

## Active Smoke/Inferno Intervals

- `10.0s` - `106.5s`, rows `194`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `104.5`, LSTM `0.4509`, XGBoost `0.6672`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2886`, XGBoost `0.4951`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.2591`, XGBoost `0.4613`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3061`, XGBoost `0.5022`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.2943`, XGBoost `0.4762`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.3041`, XGBoost `0.4762`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.2882`, XGBoost `0.4602`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.3986`, XGBoost `0.5642`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3309`, XGBoost `0.4853`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3430`, XGBoost `0.4951`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
