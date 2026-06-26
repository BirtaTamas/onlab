# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `6`
- rows: `263`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 263 | 1.000 | 0.491844 | 0.585823 | -0.093979 | 0 | 263 | 0.220532 | 0.768061 |
| active/recent utility | 263 | 1.000 | 0.491844 | 0.585823 | -0.093979 | 0 | 263 | 0.220532 | 0.768061 |
| strong utility action | 215 | 0.817 | 0.425109 | 0.529988 | -0.104879 | 0 | 215 | 0.106977 | 0.716279 |
| utility damage | 10 | 0.038 | 0.456136 | 0.525181 | -0.069045 | 0 | 10 | 0.000000 | 1.000000 |
| active smoke/inferno | 206 | 0.783 | 0.423764 | 0.530226 | -0.106462 | 0 | 206 | 0.111650 | 0.703883 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 263 | 1.000 | 0.491844 | 0.585823 | -0.093979 | 0 | 263 | 0.220532 | 0.768061 |

## Active Smoke/Inferno Intervals

- `2.5s` - `40.0s`, rows `76`
- `46.0s` - `53.5s`, rows `16`
- `57.0s` - `108.0s`, rows `103`
- `109.5s` - `114.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `106.0`, LSTM `0.4019`, XGBoost `0.7546`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.4257`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.4525`, XGBoost `0.7513`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.4082`, XGBoost `0.6913`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.2474`, XGBoost `0.5168`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.4836`, XGBoost `0.7443`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.4886`, XGBoost `0.7492`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.5046`, XGBoost `0.7633`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.4947`, XGBoost `0.7527`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.2683`, XGBoost `0.5138`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
