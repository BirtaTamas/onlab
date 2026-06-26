# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `10`
- rows: `169`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.462678 | 0.463398 | -0.000720 | 82 | 87 | 0.218935 | 0.213018 |
| active/recent utility | 169 | 1.000 | 0.462678 | 0.463398 | -0.000720 | 82 | 87 | 0.218935 | 0.213018 |
| strong utility action | 149 | 0.882 | 0.465121 | 0.464241 | 0.000879 | 78 | 71 | 0.248322 | 0.241611 |
| utility damage | 22 | 0.130 | 0.360712 | 0.331346 | 0.029367 | 17 | 5 | 0.000000 | 0.000000 |
| active smoke/inferno | 149 | 0.882 | 0.465121 | 0.464241 | 0.000879 | 78 | 71 | 0.248322 | 0.241611 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 169 | 1.000 | 0.462678 | 0.463398 | -0.000720 | 82 | 87 | 0.218935 | 0.213018 |

## Active Smoke/Inferno Intervals

- `10.0s` - `84.0s`, rows `149`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.5937`, XGBoost `0.8779`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5918`, XGBoost `0.8713`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.6095`, XGBoost `0.8713`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.6177`, XGBoost `0.8713`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.6576`, XGBoost `0.8740`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4057`, XGBoost `0.2009`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.6741`, XGBoost `0.8779`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.3943`, XGBoost `0.2009`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.3722`, XGBoost `0.1844`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7583`, XGBoost `0.9350`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
