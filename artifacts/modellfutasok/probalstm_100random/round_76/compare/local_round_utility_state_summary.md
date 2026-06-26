# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `19`
- rows: `208`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.380477 | 0.453188 | -0.072711 | 208 | 0 | 0.615385 | 0.221154 |
| active/recent utility | 208 | 1.000 | 0.380477 | 0.453188 | -0.072711 | 208 | 0 | 0.615385 | 0.221154 |
| strong utility action | 193 | 0.928 | 0.378876 | 0.451461 | -0.072584 | 193 | 0 | 0.585492 | 0.227979 |
| utility damage | 10 | 0.048 | 0.511335 | 0.570696 | -0.059362 | 10 | 0 | 0.400000 | 0.000000 |
| active smoke/inferno | 193 | 0.928 | 0.378876 | 0.451461 | -0.072584 | 193 | 0 | 0.585492 | 0.227979 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 208 | 1.000 | 0.380477 | 0.453188 | -0.072711 | 208 | 0 | 0.615385 | 0.221154 |

## Active Smoke/Inferno Intervals

- `6.5s` - `102.5s`, rows `193`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `84.5`, LSTM `0.0926`, XGBoost `0.4305`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2918`, XGBoost `0.5461`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.0319`, XGBoost `0.2751`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.0312`, XGBoost `0.2739`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0337`, XGBoost `0.2739`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3080`, XGBoost `0.5469`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0437`, XGBoost `0.2739`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.3189`, XGBoost `0.5406`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.3258`, XGBoost `0.5452`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3376`, XGBoost `0.5523`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
