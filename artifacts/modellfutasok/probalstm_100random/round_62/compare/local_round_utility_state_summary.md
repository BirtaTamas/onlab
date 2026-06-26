# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `16`
- rows: `166`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 166 | 1.000 | 0.270967 | 0.343244 | -0.072277 | 150 | 16 | 0.903614 | 0.765060 |
| active/recent utility | 166 | 1.000 | 0.270967 | 0.343244 | -0.072277 | 150 | 16 | 0.903614 | 0.765060 |
| strong utility action | 100 | 0.602 | 0.284264 | 0.334101 | -0.049838 | 84 | 16 | 0.840000 | 0.680000 |
| utility damage | 20 | 0.120 | 0.538568 | 0.560413 | -0.021845 | 11 | 9 | 0.300000 | 0.000000 |
| active smoke/inferno | 90 | 0.542 | 0.266434 | 0.313282 | -0.046848 | 74 | 16 | 0.822222 | 0.755556 |
| recent utility last 5s | 10 | 0.060 | 0.444730 | 0.521473 | -0.076743 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 166 | 1.000 | 0.270967 | 0.343244 | -0.072277 | 150 | 16 | 0.903614 | 0.765060 |

## Active Smoke/Inferno Intervals

- `8.5s` - `39.0s`, rows `62`
- `69.0s` - `82.5s`, rows `28`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.1523`, XGBoost `0.3055`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1749`, XGBoost `0.3250`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1638`, XGBoost `0.3091`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4576`, XGBoost `0.5965`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `66.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1724`, XGBoost `0.3109`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1650`, XGBoost `0.3027`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4588`, XGBoost `0.5965`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `73.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1736`, XGBoost `0.3109`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1706`, XGBoost `0.3027`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1860`, XGBoost `0.3162`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
