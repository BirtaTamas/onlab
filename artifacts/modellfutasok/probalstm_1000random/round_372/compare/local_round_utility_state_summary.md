# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `5`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.244560 | 0.292586 | -0.048026 | 132 | 10 | 0.852113 | 0.880282 |
| active/recent utility | 142 | 1.000 | 0.244560 | 0.292586 | -0.048026 | 132 | 10 | 0.852113 | 0.880282 |
| strong utility action | 113 | 0.796 | 0.251342 | 0.300111 | -0.048769 | 103 | 10 | 0.814159 | 0.849558 |
| utility damage | 16 | 0.113 | 0.280731 | 0.313365 | -0.032633 | 12 | 4 | 0.875000 | 1.000000 |
| active smoke/inferno | 107 | 0.754 | 0.264794 | 0.312998 | -0.048204 | 97 | 10 | 0.803738 | 0.841121 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.244560 | 0.292586 | -0.048026 | 132 | 10 | 0.852113 | 0.880282 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.5s`, rows `93`
- `56.5s` - `63.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.2748`, XGBoost `0.4556`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0298`, XGBoost `0.2035`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0344`, XGBoost `0.2035`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.2901`, XGBoost `0.4556`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0432`, XGBoost `0.2035`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0137`, XGBoost `0.1700`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0527`, XGBoost `0.2038`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0544`, XGBoost `0.2034`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.3121`, XGBoost `0.4567`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6410`, XGBoost `0.7818`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
