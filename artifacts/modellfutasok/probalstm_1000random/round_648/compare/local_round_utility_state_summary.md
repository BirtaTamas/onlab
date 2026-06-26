# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `8`
- rows: `124`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 124 | 1.000 | 0.301871 | 0.331105 | -0.029234 | 94 | 30 | 0.870968 | 0.677419 |
| active/recent utility | 124 | 1.000 | 0.301871 | 0.331105 | -0.029234 | 94 | 30 | 0.870968 | 0.677419 |
| strong utility action | 116 | 0.935 | 0.305481 | 0.335299 | -0.029817 | 86 | 30 | 0.870690 | 0.689655 |
| utility damage | 37 | 0.298 | 0.268586 | 0.287054 | -0.018468 | 26 | 11 | 0.810811 | 0.648649 |
| active smoke/inferno | 106 | 0.855 | 0.287435 | 0.317583 | -0.030148 | 76 | 30 | 0.886792 | 0.754717 |
| recent utility last 5s | 10 | 0.081 | 0.496772 | 0.523085 | -0.026313 | 10 | 0 | 0.700000 | 0.000000 |
| flash effect present | 124 | 1.000 | 0.301871 | 0.331105 | -0.029234 | 94 | 30 | 0.870968 | 0.677419 |

## Active Smoke/Inferno Intervals

- `7.0s` - `59.5s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.5`, LSTM `0.3941`, XGBoost `0.5555`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.2231`, XGBoost `0.3770`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4275`, XGBoost `0.5795`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3933`, XGBoost `0.5284`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.3967`, XGBoost `0.5284`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.4514`, XGBoost `0.5825`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4467`, XGBoost `0.5775`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4492`, XGBoost `0.5795`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2420`, XGBoost `0.3667`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.2532`, XGBoost `0.3759`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
