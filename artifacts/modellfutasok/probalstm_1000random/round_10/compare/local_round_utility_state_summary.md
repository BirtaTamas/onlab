# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-rare-atom-bo3-DWQZo2y3LVjgpuOkyCDf4V/3dmax-vs-rare-atom-m2-ancient.csv`
- round_num: `4`
- rows: `155`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 155 | 1.000 | 0.603717 | 0.578263 | 0.025454 | 116 | 39 | 0.832258 | 0.406452 |
| active/recent utility | 155 | 1.000 | 0.603717 | 0.578263 | 0.025454 | 116 | 39 | 0.832258 | 0.406452 |
| strong utility action | 151 | 0.974 | 0.604880 | 0.580123 | 0.024757 | 112 | 39 | 0.827815 | 0.390728 |
| utility damage | 10 | 0.065 | 0.883116 | 0.901844 | -0.018728 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 142 | 0.916 | 0.608011 | 0.584564 | 0.023447 | 103 | 39 | 0.816901 | 0.352113 |
| recent utility last 5s | 10 | 0.065 | 0.555671 | 0.509815 | 0.045856 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 155 | 1.000 | 0.603717 | 0.578263 | 0.025454 | 116 | 39 | 0.832258 | 0.406452 |

## Active Smoke/Inferno Intervals

- `6.5s` - `77.0s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.4752`, XGBoost `0.2837`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.6470`, XGBoost `0.7913`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.4404`, XGBoost `0.3086`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8417`, XGBoost `0.9258`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5514`, XGBoost `0.4776`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5467`, XGBoost `0.4760`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5440`, XGBoost `0.4760`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5465`, XGBoost `0.4788`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.5736`, XGBoost `0.5070`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5457`, XGBoost `0.4793`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
