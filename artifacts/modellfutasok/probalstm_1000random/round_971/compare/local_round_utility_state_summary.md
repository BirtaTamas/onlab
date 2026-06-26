# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `14`
- rows: `260`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 260 | 1.000 | 0.402810 | 0.425281 | -0.022471 | 100 | 160 | 0.265385 | 0.276923 |
| active/recent utility | 260 | 1.000 | 0.402810 | 0.425281 | -0.022471 | 100 | 160 | 0.265385 | 0.276923 |
| strong utility action | 179 | 0.688 | 0.312202 | 0.322587 | -0.010386 | 80 | 99 | 0.122905 | 0.117318 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 169 | 0.650 | 0.313220 | 0.323376 | -0.010156 | 77 | 92 | 0.130178 | 0.124260 |
| recent utility last 5s | 10 | 0.038 | 0.294988 | 0.309259 | -0.014271 | 3 | 7 | 0.000000 | 0.000000 |
| flash effect present | 260 | 1.000 | 0.402810 | 0.425281 | -0.022471 | 100 | 160 | 0.265385 | 0.276923 |

## Active Smoke/Inferno Intervals

- `10.5s` - `89.0s`, rows `158`
- `122.0s` - `127.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `124.0`, LSTM `0.6452`, XGBoost `0.8393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `123.0`, LSTM `0.6593`, XGBoost `0.8393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4596`, XGBoost `0.3033`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `122.5`, LSTM `0.6837`, XGBoost `0.8393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `124.5`, LSTM `0.6889`, XGBoost `0.8372`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4105`, XGBoost `0.2768`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.3702`, XGBoost `0.2391`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1685`, XGBoost `0.2989`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.4022`, XGBoost `0.2737`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `123.5`, LSTM `0.7125`, XGBoost `0.8393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
