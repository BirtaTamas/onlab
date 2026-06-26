# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `22`
- rows: `248`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 248 | 1.000 | 0.551472 | 0.438718 | 0.112754 | 68 | 180 | 0.286290 | 0.495968 |
| active/recent utility | 248 | 1.000 | 0.551472 | 0.438718 | 0.112754 | 68 | 180 | 0.286290 | 0.495968 |
| strong utility action | 161 | 0.649 | 0.524158 | 0.431394 | 0.092764 | 60 | 101 | 0.236025 | 0.465839 |
| utility damage | 20 | 0.081 | 0.501468 | 0.434920 | 0.066548 | 1 | 19 | 0.400000 | 0.400000 |
| active smoke/inferno | 161 | 0.649 | 0.524158 | 0.431394 | 0.092764 | 60 | 101 | 0.236025 | 0.465839 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 248 | 1.000 | 0.551472 | 0.438718 | 0.112754 | 68 | 180 | 0.286290 | 0.495968 |

## Active Smoke/Inferno Intervals

- `8.0s` - `68.5s`, rows `122`
- `73.0s` - `79.5s`, rows `14`
- `86.5s` - `93.0s`, rows `14`
- `105.0s` - `110.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `109.5`, LSTM `0.8164`, XGBoost `0.5079`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.8023`, XGBoost `0.5085`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.7954`, XGBoost `0.5035`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.5112`, XGBoost `0.2459`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.5079`, XGBoost `0.2459`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.5047`, XGBoost `0.2459`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.5034`, XGBoost `0.2459`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.5005`, XGBoost `0.2459`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.7630`, XGBoost `0.5085`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.4833`, XGBoost `0.2300`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
