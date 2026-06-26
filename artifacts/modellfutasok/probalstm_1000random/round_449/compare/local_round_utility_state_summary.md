# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `10`
- rows: `219`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 219 | 1.000 | 0.127579 | 0.197009 | -0.069430 | 219 | 0 | 0.990868 | 0.767123 |
| active/recent utility | 219 | 1.000 | 0.127579 | 0.197009 | -0.069430 | 219 | 0 | 0.990868 | 0.767123 |
| strong utility action | 211 | 0.963 | 0.128445 | 0.199469 | -0.071024 | 211 | 0 | 0.990521 | 0.767773 |
| utility damage | 10 | 0.046 | 0.340166 | 0.445601 | -0.105435 | 10 | 0 | 0.800000 | 0.400000 |
| active smoke/inferno | 197 | 0.900 | 0.106830 | 0.176676 | -0.069846 | 197 | 0 | 0.989848 | 0.822335 |
| recent utility last 5s | 24 | 0.110 | 0.412637 | 0.524236 | -0.111599 | 24 | 0 | 0.958333 | 0.000000 |
| flash effect present | 219 | 1.000 | 0.127579 | 0.197009 | -0.069430 | 219 | 0 | 0.990868 | 0.767123 |

## Active Smoke/Inferno Intervals

- `8.0s` - `75.0s`, rows `135`
- `78.5s` - `109.0s`, rows `62`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.2996`, XGBoost `0.5087`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3141`, XGBoost `0.5155`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3330`, XGBoost `0.5218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3209`, XGBoost `0.5087`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3232`, XGBoost `0.5087`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1085`, XGBoost `0.2819`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3692`, XGBoost `0.5382`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `23.0`, recent_utility `1`
- seconds `22.0`, LSTM `0.3527`, XGBoost `0.5198`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `21.5`, LSTM `0.3509`, XGBoost `0.5147`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `60.5`, LSTM `0.0453`, XGBoost `0.2075`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
