# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `8`
- rows: `166`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 166 | 1.000 | 0.128357 | 0.178279 | -0.049922 | 166 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 166 | 1.000 | 0.128357 | 0.178279 | -0.049922 | 166 | 0 | 1.000000 | 1.000000 |
| strong utility action | 123 | 0.741 | 0.143564 | 0.202788 | -0.059225 | 123 | 0 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.066 | 0.083962 | 0.173578 | -0.089616 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 113 | 0.681 | 0.125337 | 0.177010 | -0.051674 | 113 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.060 | 0.349530 | 0.494081 | -0.144550 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 166 | 1.000 | 0.128357 | 0.178279 | -0.049922 | 166 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `65.5s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.3228`, XGBoost `0.4928`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.3287`, XGBoost `0.4928`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.3307`, XGBoost `0.4937`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `26.0`, LSTM `0.1489`, XGBoost `0.3079`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.3385`, XGBoost `0.4959`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.3431`, XGBoost `0.4928`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `34.0`, LSTM `0.0643`, XGBoost `0.2083`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `27.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.3532`, XGBoost `0.4946`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.3534`, XGBoost `0.4946`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `17.5`, LSTM `0.3617`, XGBoost `0.4967`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
