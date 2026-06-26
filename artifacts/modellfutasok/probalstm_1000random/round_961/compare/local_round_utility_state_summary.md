# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `6`
- rows: `210`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.194889 | 0.251030 | -0.056141 | 210 | 0 | 1.000000 | 0.961905 |
| active/recent utility | 210 | 1.000 | 0.194889 | 0.251030 | -0.056141 | 210 | 0 | 1.000000 | 0.961905 |
| strong utility action | 129 | 0.614 | 0.272753 | 0.341326 | -0.068573 | 129 | 0 | 1.000000 | 0.937984 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 129 | 0.614 | 0.272753 | 0.341326 | -0.068573 | 129 | 0 | 1.000000 | 0.937984 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 210 | 1.000 | 0.194889 | 0.251030 | -0.056141 | 210 | 0 | 1.000000 | 0.961905 |

## Active Smoke/Inferno Intervals

- `7.5s` - `71.5s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.3021`, XGBoost `0.4790`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3121`, XGBoost `0.4789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3129`, XGBoost `0.4789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.3142`, XGBoost `0.4790`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.3143`, XGBoost `0.4789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3159`, XGBoost `0.4789`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.3508`, XGBoost `0.5096`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3322`, XGBoost `0.4790`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.3335`, XGBoost `0.4802`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.3351`, XGBoost `0.4790`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
