# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `14`
- rows: `159`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 159 | 1.000 | 0.071741 | 0.107980 | -0.036239 | 159 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 159 | 1.000 | 0.071741 | 0.107980 | -0.036239 | 159 | 0 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.667 | 0.103738 | 0.155295 | -0.051557 | 106 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 96 | 0.604 | 0.090902 | 0.140420 | -0.049518 | 96 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.126 | 0.217766 | 0.304074 | -0.086308 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 159 | 1.000 | 0.071741 | 0.107980 | -0.036239 | 159 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `53.5s`, rows `96`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.2225`, XGBoost `0.3765`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2269`, XGBoost `0.3717`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1379`, XGBoost `0.2767`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1674`, XGBoost `0.2985`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1503`, XGBoost `0.2813`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2493`, XGBoost `0.3784`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1719`, XGBoost `0.2984`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1740`, XGBoost `0.2989`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1891`, XGBoost `0.3113`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `1`
- seconds `16.5`, LSTM `0.1902`, XGBoost `0.3107`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `1`
