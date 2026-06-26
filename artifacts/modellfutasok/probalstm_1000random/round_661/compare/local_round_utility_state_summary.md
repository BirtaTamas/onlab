# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `16`
- rows: `188`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.531242 | 0.634299 | -0.103058 | 13 | 175 | 0.569149 | 0.984043 |
| active/recent utility | 188 | 1.000 | 0.531242 | 0.634299 | -0.103058 | 13 | 175 | 0.569149 | 0.984043 |
| strong utility action | 169 | 0.899 | 0.539792 | 0.645321 | -0.105530 | 13 | 156 | 0.633136 | 0.982249 |
| utility damage | 10 | 0.053 | 0.491615 | 0.591531 | -0.099915 | 0 | 10 | 0.000000 | 1.000000 |
| active smoke/inferno | 169 | 0.899 | 0.539792 | 0.645321 | -0.105530 | 13 | 156 | 0.633136 | 0.982249 |
| recent utility last 5s | 20 | 0.106 | 0.489350 | 0.657368 | -0.168017 | 0 | 20 | 0.050000 | 1.000000 |
| flash effect present | 188 | 1.000 | 0.531242 | 0.634299 | -0.103058 | 13 | 175 | 0.569149 | 0.984043 |

## Active Smoke/Inferno Intervals

- `9.5s` - `93.5s`, rows `169`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.4982`, XGBoost `0.6822`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5014`, XGBoost `0.6852`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.4833`, XGBoost `0.6655`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5016`, XGBoost `0.6832`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4760`, XGBoost `0.6573`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `33.0`, LSTM `0.4843`, XGBoost `0.6655`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4859`, XGBoost `0.6666`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `33.5`, LSTM `0.4878`, XGBoost `0.6663`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5028`, XGBoost `0.6812`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4871`, XGBoost `0.6655`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
