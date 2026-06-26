# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `8`
- rows: `134`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.644072 | 0.729251 | -0.085179 | 0 | 134 | 0.679104 | 0.701493 |
| active/recent utility | 134 | 1.000 | 0.644072 | 0.729251 | -0.085179 | 0 | 134 | 0.679104 | 0.701493 |
| strong utility action | 120 | 0.896 | 0.668556 | 0.753170 | -0.084613 | 0 | 120 | 0.750000 | 0.775000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 120 | 0.896 | 0.668556 | 0.753170 | -0.084613 | 0 | 120 | 0.750000 | 0.775000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 134 | 1.000 | 0.644072 | 0.729251 | -0.085179 | 0 | 134 | 0.679104 | 0.701493 |

## Active Smoke/Inferno Intervals

- `6.5s` - `66.0s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.5344`, XGBoost `0.7399`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5402`, XGBoost `0.7398`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5403`, XGBoost `0.7385`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5386`, XGBoost `0.7360`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5413`, XGBoost `0.7385`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5465`, XGBoost `0.7384`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5474`, XGBoost `0.7387`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5485`, XGBoost `0.7377`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5285`, XGBoost `0.7153`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5545`, XGBoost `0.7394`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
