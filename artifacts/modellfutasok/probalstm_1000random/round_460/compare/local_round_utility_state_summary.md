# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.730500 | 0.755625 | -0.025125 | 41 | 189 | 0.960870 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.730500 | 0.755625 | -0.025125 | 41 | 189 | 0.960870 | 1.000000 |
| strong utility action | 188 | 0.817 | 0.712330 | 0.742291 | -0.029962 | 27 | 161 | 0.952128 | 1.000000 |
| utility damage | 18 | 0.078 | 0.507507 | 0.539057 | -0.031550 | 0 | 18 | 0.611111 | 1.000000 |
| active smoke/inferno | 188 | 0.817 | 0.712330 | 0.742291 | -0.029962 | 27 | 161 | 0.952128 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.730500 | 0.755625 | -0.025125 | 41 | 189 | 0.960870 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `78.0s`, rows `144`
- `81.5s` - `103.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.8120`, XGBoost `0.9293`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8150`, XGBoost `0.9308`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6234`, XGBoost `0.7386`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.8154`, XGBoost `0.9286`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.8242`, XGBoost `0.9293`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.8267`, XGBoost `0.9268`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.8347`, XGBoost `0.9310`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6428`, XGBoost `0.7385`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.8337`, XGBoost `0.9286`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.8371`, XGBoost `0.9278`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
