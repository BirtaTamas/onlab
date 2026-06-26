# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `9`
- rows: `133`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.702548 | 0.701794 | 0.000754 | 64 | 69 | 1.000000 | 1.000000 |
| active/recent utility | 133 | 1.000 | 0.702548 | 0.701794 | 0.000754 | 64 | 69 | 1.000000 | 1.000000 |
| strong utility action | 121 | 0.910 | 0.689841 | 0.688095 | 0.001746 | 60 | 61 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 111 | 0.835 | 0.695317 | 0.697055 | -0.001738 | 50 | 61 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.150 | 0.792091 | 0.785010 | 0.007081 | 10 | 10 | 1.000000 | 1.000000 |
| flash effect present | 133 | 1.000 | 0.702548 | 0.701794 | 0.000754 | 64 | 69 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `62.0s`, rows `111`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.6122`, XGBoost `0.7403`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6254`, XGBoost `0.7403`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6261`, XGBoost `0.7403`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.6264`, XGBoost `0.7403`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6295`, XGBoost `0.7403`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6413`, XGBoost `0.7403`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6726`, XGBoost `0.5738`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6425`, XGBoost `0.7399`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6694`, XGBoost `0.5738`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6513`, XGBoost `0.5649`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
