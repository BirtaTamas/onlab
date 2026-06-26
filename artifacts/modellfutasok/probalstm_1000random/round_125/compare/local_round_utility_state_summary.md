# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `4`
- rows: `123`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.555036 | 0.616952 | -0.061915 | 7 | 116 | 0.422764 | 0.552846 |
| active/recent utility | 123 | 1.000 | 0.555036 | 0.616952 | -0.061915 | 7 | 116 | 0.422764 | 0.552846 |
| strong utility action | 108 | 0.878 | 0.577783 | 0.631373 | -0.053590 | 7 | 101 | 0.481481 | 0.490741 |
| utility damage | 20 | 0.163 | 0.608202 | 0.642783 | -0.034581 | 2 | 18 | 0.400000 | 0.400000 |
| active smoke/inferno | 108 | 0.878 | 0.577783 | 0.631373 | -0.053590 | 7 | 101 | 0.481481 | 0.490741 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 123 | 1.000 | 0.555036 | 0.616952 | -0.061915 | 7 | 116 | 0.422764 | 0.552846 |

## Active Smoke/Inferno Intervals

- `7.5s` - `61.0s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.0`, LSTM `0.1214`, XGBoost `0.4104`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1493`, XGBoost `0.4045`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.2461`, XGBoost `0.4513`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2102`, XGBoost `0.4060`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0645`, XGBoost `0.2233`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3397`, XGBoost `0.4921`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0890`, XGBoost `0.2367`, closer `xgboost`, smoke `6`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3471`, XGBoost `0.4921`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0847`, XGBoost `0.2245`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.3685`, XGBoost `0.5032`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
