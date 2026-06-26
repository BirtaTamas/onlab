# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX/flyquest-vs-fluxo-ancient.csv`
- round_num: `2`
- rows: `121`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 121 | 1.000 | 0.940365 | 0.978181 | -0.037817 | 0 | 121 | 1.000000 | 1.000000 |
| active/recent utility | 121 | 1.000 | 0.940365 | 0.978181 | -0.037817 | 0 | 121 | 1.000000 | 1.000000 |
| strong utility action | 101 | 0.835 | 0.940429 | 0.977170 | -0.036741 | 0 | 101 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 101 | 0.835 | 0.940429 | 0.977170 | -0.036741 | 0 | 101 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 121 | 1.000 | 0.940365 | 0.978181 | -0.037817 | 0 | 121 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `57.5s`, rows `101`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.9296`, XGBoost `0.9787`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.9239`, XGBoost `0.9722`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.9241`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.9251`, XGBoost `0.9720`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.9335`, XGBoost `0.9800`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.9327`, XGBoost `0.9789`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.9330`, XGBoost `0.9789`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.9329`, XGBoost `0.9787`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.9339`, XGBoost `0.9793`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.9264`, XGBoost `0.9717`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
