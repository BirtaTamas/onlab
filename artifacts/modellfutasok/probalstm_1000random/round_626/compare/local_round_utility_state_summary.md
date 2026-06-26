# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `4`
- rows: `265`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 265 | 1.000 | 0.244937 | 0.259719 | -0.014782 | 192 | 73 | 0.920755 | 0.807547 |
| active/recent utility | 265 | 1.000 | 0.244937 | 0.259719 | -0.014782 | 192 | 73 | 0.920755 | 0.807547 |
| strong utility action | 201 | 0.758 | 0.270363 | 0.283380 | -0.013017 | 128 | 73 | 0.895522 | 0.855721 |
| utility damage | 26 | 0.098 | 0.323720 | 0.295614 | 0.028106 | 7 | 19 | 1.000000 | 1.000000 |
| active smoke/inferno | 201 | 0.758 | 0.270363 | 0.283380 | -0.013017 | 128 | 73 | 0.895522 | 0.855721 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 265 | 1.000 | 0.244937 | 0.259719 | -0.014782 | 192 | 73 | 0.920755 | 0.807547 |

## Active Smoke/Inferno Intervals

- `11.0s` - `93.5s`, rows `166`
- `115.0s` - `132.0s`, rows `35`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.5642`, XGBoost `0.3362`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5367`, XGBoost `0.3444`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.4845`, XGBoost `0.2975`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5685`, XGBoost `0.3872`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5493`, XGBoost `0.3842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.3388`, XGBoost `0.4953`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.3442`, XGBoost `0.4953`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5295`, XGBoost `0.3873`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3674`, XGBoost `0.4953`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4042`, XGBoost `0.2815`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
