# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `12`
- rows: `135`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.188487 | 0.311680 | -0.123194 | 135 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 135 | 1.000 | 0.188487 | 0.311680 | -0.123194 | 135 | 0 | 1.000000 | 1.000000 |
| strong utility action | 118 | 0.874 | 0.170902 | 0.285719 | -0.114818 | 118 | 0 | 1.000000 | 1.000000 |
| utility damage | 9 | 0.067 | 0.003089 | 0.017013 | -0.013924 | 9 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 118 | 0.874 | 0.170902 | 0.285719 | -0.114818 | 118 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.188487 | 0.311680 | -0.123194 | 135 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `67.0s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.0`, LSTM `0.0539`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0574`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0687`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0705`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0803`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0829`, XGBoost `0.4217`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.1199`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1317`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1404`, XGBoost `0.4483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1500`, XGBoost `0.4499`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
