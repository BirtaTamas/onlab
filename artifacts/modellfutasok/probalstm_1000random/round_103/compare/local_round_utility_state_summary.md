# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `16`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.548542 | 0.608289 | -0.059747 | 33 | 124 | 0.515924 | 0.585987 |
| active/recent utility | 157 | 1.000 | 0.548542 | 0.608289 | -0.059747 | 33 | 124 | 0.515924 | 0.585987 |
| strong utility action | 147 | 0.936 | 0.530666 | 0.589955 | -0.059290 | 32 | 115 | 0.482993 | 0.564626 |
| utility damage | 20 | 0.127 | 0.484305 | 0.492394 | -0.008090 | 5 | 15 | 0.200000 | 0.050000 |
| active smoke/inferno | 134 | 0.854 | 0.526965 | 0.599253 | -0.072289 | 19 | 115 | 0.432836 | 0.619403 |
| recent utility last 5s | 16 | 0.102 | 0.551162 | 0.494370 | 0.056792 | 13 | 3 | 0.812500 | 0.000000 |
| flash effect present | 157 | 1.000 | 0.548542 | 0.608289 | -0.059747 | 33 | 124 | 0.515924 | 0.585987 |

## Active Smoke/Inferno Intervals

- `7.0s` - `73.5s`, rows `134`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.1634`, XGBoost `0.4948`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.3755`, XGBoost `0.6495`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.2244`, XGBoost `0.4953`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3824`, XGBoost `0.6480`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3989`, XGBoost `0.6525`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.2886`, XGBoost `0.4940`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7223`, XGBoost `0.9172`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.7166`, XGBoost `0.9018`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.7214`, XGBoost `0.9027`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7226`, XGBoost `0.9027`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
