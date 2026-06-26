# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `23`
- rows: `106`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 106 | 1.000 | 0.506882 | 0.511801 | -0.004919 | 40 | 66 | 0.698113 | 0.584906 |
| active/recent utility | 106 | 1.000 | 0.506882 | 0.511801 | -0.004919 | 40 | 66 | 0.698113 | 0.584906 |
| strong utility action | 102 | 0.962 | 0.505731 | 0.508256 | -0.002525 | 40 | 62 | 0.686275 | 0.568627 |
| utility damage | 11 | 0.104 | 0.606852 | 0.654252 | -0.047400 | 3 | 8 | 0.909091 | 0.727273 |
| active smoke/inferno | 92 | 0.868 | 0.498432 | 0.497881 | 0.000550 | 40 | 52 | 0.652174 | 0.521739 |
| recent utility last 5s | 10 | 0.094 | 0.572882 | 0.603704 | -0.030822 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 106 | 1.000 | 0.506882 | 0.511801 | -0.004919 | 40 | 66 | 0.698113 | 0.584906 |

## Active Smoke/Inferno Intervals

- `7.0s` - `52.5s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.4533`, XGBoost `0.2807`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4403`, XGBoost `0.2820`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4300`, XGBoost `0.2755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6848`, XGBoost `0.8382`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.4353`, XGBoost `0.2820`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7756`, XGBoost `0.9288`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `53.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4259`, XGBoost `0.2820`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.4187`, XGBoost `0.2818`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.4183`, XGBoost `0.2815`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4182`, XGBoost `0.2815`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
