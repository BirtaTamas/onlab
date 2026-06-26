# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `13`
- rows: `199`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.666440 | 0.769398 | -0.102958 | 46 | 153 | 0.839196 | 0.768844 |
| active/recent utility | 199 | 1.000 | 0.666440 | 0.769398 | -0.102958 | 46 | 153 | 0.839196 | 0.768844 |
| strong utility action | 107 | 0.538 | 0.568833 | 0.640928 | -0.072095 | 43 | 64 | 0.794393 | 0.598131 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 97 | 0.487 | 0.574459 | 0.655024 | -0.080565 | 38 | 59 | 0.773196 | 0.608247 |
| recent utility last 5s | 10 | 0.050 | 0.514261 | 0.504196 | 0.010066 | 5 | 5 | 1.000000 | 0.500000 |
| flash effect present | 199 | 1.000 | 0.666440 | 0.769398 | -0.102958 | 46 | 153 | 0.839196 | 0.768844 |

## Active Smoke/Inferno Intervals

- `7.0s` - `33.0s`, rows `53`
- `42.5s` - `64.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.3861`, XGBoost `0.9199`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.4030`, XGBoost `0.9195`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4115`, XGBoost `0.9195`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.4525`, XGBoost `0.9205`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.4819`, XGBoost `0.9205`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5268`, XGBoost `0.9200`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5512`, XGBoost `0.9198`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5462`, XGBoost `0.8755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5987`, XGBoost `0.8755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0735`, XGBoost `0.2375`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
