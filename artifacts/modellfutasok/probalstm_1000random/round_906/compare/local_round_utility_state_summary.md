# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `4`
- rows: `129`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 129 | 1.000 | 0.848069 | 0.892196 | -0.044127 | 0 | 129 | 0.844961 | 1.000000 |
| active/recent utility | 129 | 1.000 | 0.848069 | 0.892196 | -0.044127 | 0 | 129 | 0.844961 | 1.000000 |
| strong utility action | 108 | 0.837 | 0.875316 | 0.922402 | -0.047086 | 0 | 108 | 0.879630 | 1.000000 |
| utility damage | 11 | 0.085 | 0.987017 | 0.997729 | -0.010711 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 103 | 0.798 | 0.869883 | 0.918742 | -0.048859 | 0 | 103 | 0.873786 | 1.000000 |
| recent utility last 5s | 10 | 0.078 | 0.787531 | 0.895267 | -0.107736 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 129 | 1.000 | 0.848069 | 0.892196 | -0.044127 | 0 | 129 | 0.844961 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `57.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.5642`, XGBoost `0.7428`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `3.0`, recent_utility `1`
- seconds `13.5`, LSTM `0.5831`, XGBoost `0.7428`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `3.0`, recent_utility `1`
- seconds `12.5`, LSTM `0.5872`, XGBoost `0.7433`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.7665`, XGBoost `0.9021`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `3.0`, recent_utility `1`
- seconds `15.5`, LSTM `0.7808`, XGBoost `0.9020`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `3.0`, recent_utility `1`
- seconds `12.0`, LSTM `0.4781`, XGBoost `0.5956`, closer `xgboost`, smoke `5`, inferno `4`, utility_damage `3.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.7892`, XGBoost `0.9026`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `3.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.7947`, XGBoost `0.9029`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `3.0`, recent_utility `1`
- seconds `16.0`, LSTM `0.8018`, XGBoost `0.9049`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `1`
- seconds `11.5`, LSTM `0.4519`, XGBoost `0.5487`, closer `xgboost`, smoke `5`, inferno `5`, utility_damage `3.0`, recent_utility `0`
