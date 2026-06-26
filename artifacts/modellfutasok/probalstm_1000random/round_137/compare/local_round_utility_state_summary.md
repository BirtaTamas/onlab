# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `8`
- rows: `155`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 155 | 1.000 | 0.656841 | 0.679176 | -0.022335 | 60 | 95 | 1.000000 | 1.000000 |
| active/recent utility | 155 | 1.000 | 0.656841 | 0.679176 | -0.022335 | 60 | 95 | 1.000000 | 1.000000 |
| strong utility action | 116 | 0.748 | 0.652518 | 0.682292 | -0.029774 | 42 | 74 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.065 | 0.573296 | 0.560657 | 0.012638 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 116 | 0.748 | 0.652518 | 0.682292 | -0.029774 | 42 | 74 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 155 | 1.000 | 0.656841 | 0.679176 | -0.022335 | 60 | 95 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `35.5s`, rows `56`
- `42.5s` - `72.0s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `55.5`, LSTM `0.5751`, XGBoost `0.7649`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.5976`, XGBoost `0.7649`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6049`, XGBoost `0.7649`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6214`, XGBoost `0.7658`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.7636`, XGBoost `0.9078`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6229`, XGBoost `0.7642`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6640`, XGBoost `0.7967`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `1.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.6355`, XGBoost `0.7649`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6694`, XGBoost `0.7967`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `1.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.8053`, XGBoost `0.9272`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
