# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `2`
- rows: `200`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.511441 | 0.605938 | -0.094497 | 2 | 198 | 0.550000 | 0.965000 |
| active/recent utility | 200 | 1.000 | 0.511441 | 0.605938 | -0.094497 | 2 | 198 | 0.550000 | 0.965000 |
| strong utility action | 154 | 0.770 | 0.467695 | 0.563884 | -0.096188 | 2 | 152 | 0.454545 | 0.954545 |
| utility damage | 20 | 0.100 | 0.440219 | 0.559202 | -0.118983 | 0 | 20 | 0.200000 | 1.000000 |
| active smoke/inferno | 152 | 0.760 | 0.468018 | 0.564154 | -0.096136 | 2 | 150 | 0.460526 | 0.953947 |
| recent utility last 5s | 10 | 0.050 | 0.472272 | 0.539167 | -0.066896 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 200 | 1.000 | 0.511441 | 0.605938 | -0.094497 | 2 | 198 | 0.550000 | 0.965000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `59.5s`, rows `108`
- `60.5s` - `82.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.5`, LSTM `0.1085`, XGBoost `0.4074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1209`, XGBoost `0.4074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1339`, XGBoost `0.4074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.1509`, XGBoost `0.4123`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.1571`, XGBoost `0.4048`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1654`, XGBoost `0.4074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3540`, XGBoost `0.5826`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3539`, XGBoost `0.5821`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3555`, XGBoost `0.5826`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1931`, XGBoost `0.4082`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
