# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `12`
- rows: `130`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 130 | 1.000 | 0.330472 | 0.392808 | -0.062336 | 84 | 46 | 0.607692 | 0.607692 |
| active/recent utility | 130 | 1.000 | 0.330472 | 0.392808 | -0.062336 | 84 | 46 | 0.607692 | 0.607692 |
| strong utility action | 117 | 0.900 | 0.281334 | 0.356300 | -0.074966 | 84 | 33 | 0.675214 | 0.675214 |
| utility damage | 20 | 0.154 | 0.399587 | 0.432972 | -0.033385 | 10 | 10 | 0.500000 | 0.500000 |
| active smoke/inferno | 117 | 0.900 | 0.281334 | 0.356300 | -0.074966 | 84 | 33 | 0.675214 | 0.675214 |
| recent utility last 5s | 10 | 0.077 | 0.745340 | 0.721086 | 0.024255 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 130 | 1.000 | 0.330472 | 0.392808 | -0.062336 | 84 | 46 | 0.607692 | 0.607692 |

## Active Smoke/Inferno Intervals

- `6.5s` - `64.5s`, rows `117`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.0878`, XGBoost `0.3446`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1048`, XGBoost `0.3432`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0624`, XGBoost `0.2922`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0620`, XGBoost `0.2783`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.1165`, XGBoost `0.3324`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0587`, XGBoost `0.2621`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1415`, XGBoost `0.3324`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0544`, XGBoost `0.2418`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0525`, XGBoost `0.2382`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0578`, XGBoost `0.2244`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
