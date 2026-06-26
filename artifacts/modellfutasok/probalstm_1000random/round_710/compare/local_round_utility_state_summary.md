# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `11`
- rows: `139`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.297382 | 0.393176 | -0.095794 | 3 | 136 | 0.179856 | 0.179856 |
| active/recent utility | 139 | 1.000 | 0.297382 | 0.393176 | -0.095794 | 3 | 136 | 0.179856 | 0.179856 |
| strong utility action | 124 | 0.892 | 0.322664 | 0.409721 | -0.087057 | 3 | 121 | 0.201613 | 0.201613 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.892 | 0.322664 | 0.409721 | -0.087057 | 3 | 121 | 0.201613 | 0.201613 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 139 | 1.000 | 0.297382 | 0.393176 | -0.095794 | 3 | 136 | 0.179856 | 0.179856 |

## Active Smoke/Inferno Intervals

- `7.5s` - `69.0s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.5`, LSTM `0.3131`, XGBoost `0.5100`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `6.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0890`, XGBoost `0.2552`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0927`, XGBoost `0.2552`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0987`, XGBoost `0.2555`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1054`, XGBoost `0.2597`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1356`, XGBoost `0.2878`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `4.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1121`, XGBoost `0.2603`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1193`, XGBoost `0.2670`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.7817`, XGBoost `0.6361`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `44.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1369`, XGBoost `0.2819`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `4.0`, recent_utility `0`
