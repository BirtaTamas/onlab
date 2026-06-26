# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `3`
- rows: `275`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 275 | 1.000 | 0.183542 | 0.267234 | -0.083691 | 272 | 3 | 0.883636 | 0.665455 |
| active/recent utility | 275 | 1.000 | 0.183542 | 0.267234 | -0.083691 | 272 | 3 | 0.883636 | 0.665455 |
| strong utility action | 106 | 0.385 | 0.393673 | 0.517928 | -0.124256 | 103 | 3 | 0.698113 | 0.292453 |
| utility damage | 10 | 0.036 | 0.457451 | 0.568224 | -0.110773 | 10 | 0 | 0.900000 | 0.000000 |
| active smoke/inferno | 106 | 0.385 | 0.393673 | 0.517928 | -0.124256 | 103 | 3 | 0.698113 | 0.292453 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 275 | 1.000 | 0.183542 | 0.267234 | -0.083691 | 272 | 3 | 0.883636 | 0.665455 |

## Active Smoke/Inferno Intervals

- `8.5s` - `61.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.1254`, XGBoost `0.4060`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1419`, XGBoost `0.4060`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1432`, XGBoost `0.4060`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1500`, XGBoost `0.4060`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1497`, XGBoost `0.4055`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1533`, XGBoost `0.4060`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0877`, XGBoost `0.3160`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.0903`, XGBoost `0.3184`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0893`, XGBoost `0.3160`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1877`, XGBoost `0.4074`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
