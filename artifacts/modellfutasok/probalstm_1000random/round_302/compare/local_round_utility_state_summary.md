# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `4`
- rows: `250`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 250 | 1.000 | 0.858314 | 0.871522 | -0.013207 | 56 | 194 | 1.000000 | 0.900000 |
| active/recent utility | 250 | 1.000 | 0.858314 | 0.871522 | -0.013207 | 56 | 194 | 1.000000 | 0.900000 |
| strong utility action | 135 | 0.540 | 0.822025 | 0.836297 | -0.014272 | 41 | 94 | 1.000000 | 0.903704 |
| utility damage | 28 | 0.112 | 0.700177 | 0.746868 | -0.046691 | 4 | 24 | 1.000000 | 1.000000 |
| active smoke/inferno | 135 | 0.540 | 0.822025 | 0.836297 | -0.014272 | 41 | 94 | 1.000000 | 0.903704 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 250 | 1.000 | 0.858314 | 0.871522 | -0.013207 | 56 | 194 | 1.000000 | 0.900000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `66.0s`, rows `121`
- `95.5s` - `102.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `102.0`, LSTM `0.7892`, XGBoost `0.9046`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6670`, XGBoost `0.7767`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6446`, XGBoost `0.7459`, closer `xgboost`, smoke `3`, inferno `5`, utility_damage `10.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6451`, XGBoost `0.7458`, closer `xgboost`, smoke `4`, inferno `5`, utility_damage `10.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.8056`, XGBoost `0.9046`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.7095`, XGBoost `0.8020`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `89.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6895`, XGBoost `0.7767`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6643`, XGBoost `0.7458`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.7036`, XGBoost `0.7802`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5661`, XGBoost `0.4921`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
