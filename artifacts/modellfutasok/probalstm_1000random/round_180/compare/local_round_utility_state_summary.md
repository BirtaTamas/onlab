# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `2`
- rows: `101`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 101 | 1.000 | 0.648816 | 0.618448 | 0.030368 | 69 | 32 | 0.950495 | 0.940594 |
| active/recent utility | 101 | 1.000 | 0.648816 | 0.618448 | 0.030368 | 69 | 32 | 0.950495 | 0.940594 |
| strong utility action | 95 | 0.941 | 0.649693 | 0.622522 | 0.027171 | 63 | 32 | 0.947368 | 0.936842 |
| utility damage | 20 | 0.198 | 0.529498 | 0.521746 | 0.007752 | 14 | 6 | 0.750000 | 0.700000 |
| active smoke/inferno | 85 | 0.842 | 0.654669 | 0.630099 | 0.024570 | 53 | 32 | 0.941176 | 0.929412 |
| recent utility last 5s | 10 | 0.099 | 0.607396 | 0.558119 | 0.049276 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 101 | 1.000 | 0.648816 | 0.618448 | 0.030368 | 69 | 32 | 0.950495 | 0.940594 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.6558`, XGBoost `0.5433`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6523`, XGBoost `0.5474`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6469`, XGBoost `0.5474`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6469`, XGBoost `0.5485`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6454`, XGBoost `0.5474`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6369`, XGBoost `0.5433`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6406`, XGBoost `0.5474`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.6449`, XGBoost `0.5524`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6392`, XGBoost `0.5474`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6348`, XGBoost `0.5433`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
