# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `1`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.812240 | 0.865209 | -0.052969 | 16 | 149 | 0.951515 | 0.975758 |
| active/recent utility | 165 | 1.000 | 0.812240 | 0.865209 | -0.052969 | 16 | 149 | 0.951515 | 0.975758 |
| strong utility action | 82 | 0.497 | 0.888386 | 0.939510 | -0.051124 | 4 | 78 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 82 | 0.497 | 0.888386 | 0.939510 | -0.051124 | 4 | 78 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.812240 | 0.865209 | -0.052969 | 16 | 149 | 0.951515 | 0.975758 |

## Active Smoke/Inferno Intervals

- `30.5s` - `71.0s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.5`, LSTM `0.6560`, XGBoost `0.8245`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6559`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6687`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6785`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6872`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.6954`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.8729`, XGBoost `0.9625`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.7349`, XGBoost `0.8243`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.8578`, XGBoost `0.9463`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.8569`, XGBoost `0.9416`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
