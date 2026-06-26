# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `15`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.268991 | 0.420498 | -0.151507 | 17 | 148 | 0.054545 | 0.418182 |
| active/recent utility | 165 | 1.000 | 0.268991 | 0.420498 | -0.151507 | 17 | 148 | 0.054545 | 0.418182 |
| strong utility action | 137 | 0.830 | 0.287733 | 0.420511 | -0.132778 | 13 | 124 | 0.065693 | 0.394161 |
| utility damage | 20 | 0.121 | 0.271350 | 0.292648 | -0.021298 | 10 | 10 | 0.150000 | 0.000000 |
| active smoke/inferno | 129 | 0.782 | 0.283695 | 0.417761 | -0.134066 | 13 | 116 | 0.069767 | 0.418605 |
| recent utility last 5s | 10 | 0.061 | 0.350630 | 0.463828 | -0.113198 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 165 | 1.000 | 0.268991 | 0.420498 | -0.151507 | 17 | 148 | 0.054545 | 0.418182 |

## Active Smoke/Inferno Intervals

- `6.0s` - `48.0s`, rows `85`
- `55.0s` - `76.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.2088`, XGBoost `0.5115`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.2176`, XGBoost `0.5115`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.2229`, XGBoost `0.5153`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2232`, XGBoost `0.5153`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2333`, XGBoost `0.5230`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2371`, XGBoost `0.5108`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.2593`, XGBoost `0.5275`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.2517`, XGBoost `0.5153`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.2563`, XGBoost `0.5153`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.2676`, XGBoost `0.5237`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
