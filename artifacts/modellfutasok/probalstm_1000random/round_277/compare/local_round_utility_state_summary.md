# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `2`
- rows: `176`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.414937 | 0.489525 | -0.074588 | 16 | 160 | 0.323864 | 0.301136 |
| active/recent utility | 176 | 1.000 | 0.414937 | 0.489525 | -0.074588 | 16 | 160 | 0.323864 | 0.301136 |
| strong utility action | 133 | 0.756 | 0.404157 | 0.483096 | -0.078939 | 13 | 120 | 0.270677 | 0.240602 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 133 | 0.756 | 0.404157 | 0.483096 | -0.078939 | 13 | 120 | 0.270677 | 0.240602 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 176 | 1.000 | 0.414937 | 0.489525 | -0.074588 | 16 | 160 | 0.323864 | 0.301136 |

## Active Smoke/Inferno Intervals

- `10.0s` - `33.5s`, rows `48`
- `45.0s` - `87.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.5`, LSTM `0.7128`, XGBoost `0.9117`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1642`, XGBoost `0.3608`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.7177`, XGBoost `0.9130`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.1374`, XGBoost `0.3288`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2952`, XGBoost `0.4863`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1410`, XGBoost `0.3309`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.1171`, XGBoost `0.3016`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1182`, XGBoost `0.3021`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3064`, XGBoost `0.4872`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.1219`, XGBoost `0.3021`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
