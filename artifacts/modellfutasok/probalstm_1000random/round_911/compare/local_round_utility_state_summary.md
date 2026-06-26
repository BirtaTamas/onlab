# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `13`
- rows: `133`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.211784 | 0.206268 | 0.005516 | 62 | 71 | 0.766917 | 0.909774 |
| active/recent utility | 133 | 1.000 | 0.211784 | 0.206268 | 0.005516 | 62 | 71 | 0.766917 | 0.909774 |
| strong utility action | 74 | 0.556 | 0.235512 | 0.229299 | 0.006213 | 37 | 37 | 0.824324 | 0.905405 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 74 | 0.556 | 0.235512 | 0.229299 | 0.006213 | 37 | 37 | 0.824324 | 0.905405 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 133 | 1.000 | 0.211784 | 0.206268 | 0.005516 | 62 | 71 | 0.766917 | 0.909774 |

## Active Smoke/Inferno Intervals

- `9.0s` - `45.5s`, rows `74`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.5180`, XGBoost `0.3995`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5116`, XGBoost `0.3967`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.3217`, XGBoost `0.2074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2789`, XGBoost `0.1672`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.4914`, XGBoost `0.3802`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.3593`, XGBoost `0.2720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.3571`, XGBoost `0.2720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.3564`, XGBoost `0.2720`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4601`, XGBoost `0.3789`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3473`, XGBoost `0.2703`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
