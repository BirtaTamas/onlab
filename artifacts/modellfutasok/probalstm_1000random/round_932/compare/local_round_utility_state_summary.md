# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `13`
- rows: `117`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.396105 | 0.512537 | -0.116432 | 1 | 116 | 0.393162 | 0.512821 |
| active/recent utility | 117 | 1.000 | 0.396105 | 0.512537 | -0.116432 | 1 | 116 | 0.393162 | 0.512821 |
| strong utility action | 46 | 0.393 | 0.181346 | 0.284370 | -0.103024 | 1 | 45 | 0.021739 | 0.086957 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 46 | 0.393 | 0.181346 | 0.284370 | -0.103024 | 1 | 45 | 0.021739 | 0.086957 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 117 | 1.000 | 0.396105 | 0.512537 | -0.116432 | 1 | 116 | 0.393162 | 0.512821 |

## Active Smoke/Inferno Intervals

- `10.5s` - `33.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.0408`, XGBoost `0.2400`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0454`, XGBoost `0.2407`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0410`, XGBoost `0.2354`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0472`, XGBoost `0.2243`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0378`, XGBoost `0.2132`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0382`, XGBoost `0.2132`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0396`, XGBoost `0.2132`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0396`, XGBoost `0.2132`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0397`, XGBoost `0.2105`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0413`, XGBoost `0.2105`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
