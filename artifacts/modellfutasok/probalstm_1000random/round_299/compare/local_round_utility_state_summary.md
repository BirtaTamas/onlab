# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `6`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.572699 | 0.611364 | -0.038665 | 31 | 134 | 0.733333 | 0.878788 |
| active/recent utility | 165 | 1.000 | 0.572699 | 0.611364 | -0.038665 | 31 | 134 | 0.733333 | 0.878788 |
| strong utility action | 141 | 0.855 | 0.582847 | 0.628031 | -0.045184 | 19 | 122 | 0.723404 | 0.858156 |
| utility damage | 32 | 0.194 | 0.594854 | 0.632712 | -0.037859 | 1 | 31 | 0.875000 | 1.000000 |
| active smoke/inferno | 136 | 0.824 | 0.585847 | 0.631981 | -0.046135 | 19 | 117 | 0.727941 | 0.852941 |
| recent utility last 5s | 10 | 0.061 | 0.487822 | 0.537891 | -0.050070 | 0 | 10 | 0.100000 | 1.000000 |
| flash effect present | 165 | 1.000 | 0.572699 | 0.611364 | -0.038665 | 31 | 134 | 0.733333 | 0.878788 |

## Active Smoke/Inferno Intervals

- `7.0s` - `37.5s`, rows `62`
- `45.5s` - `82.0s`, rows `74`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.1980`, XGBoost `0.4752`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.2397`, XGBoost `0.4754`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.2628`, XGBoost `0.4812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.2812`, XGBoost `0.4851`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.2785`, XGBoost `0.4756`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.1033`, XGBoost `0.2738`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1112`, XGBoost `0.2741`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1199`, XGBoost `0.2741`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1203`, XGBoost `0.2719`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5445`, XGBoost `0.6850`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `14.0`, recent_utility `0`
