# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `8`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.435464 | 0.512751 | -0.077287 | 9 | 221 | 0.386957 | 0.634783 |
| active/recent utility | 230 | 1.000 | 0.435464 | 0.512751 | -0.077287 | 9 | 221 | 0.386957 | 0.634783 |
| strong utility action | 167 | 0.726 | 0.465376 | 0.517403 | -0.052027 | 2 | 165 | 0.455090 | 0.742515 |
| utility damage | 40 | 0.174 | 0.493166 | 0.544002 | -0.050836 | 0 | 40 | 0.650000 | 1.000000 |
| active smoke/inferno | 157 | 0.683 | 0.465335 | 0.518847 | -0.053513 | 2 | 155 | 0.484076 | 0.789809 |
| recent utility last 5s | 10 | 0.043 | 0.466033 | 0.494730 | -0.028696 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.435464 | 0.512751 | -0.077287 | 9 | 221 | 0.386957 | 0.634783 |

## Active Smoke/Inferno Intervals

- `9.0s` - `87.0s`, rows `157`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.0`, LSTM `0.1246`, XGBoost `0.3662`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.1613`, XGBoost `0.3898`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.2058`, XGBoost `0.4166`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1015`, XGBoost `0.2722`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.3529`, XGBoost `0.5152`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.3566`, XGBoost `0.5128`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.1118`, XGBoost `0.2658`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.3672`, XGBoost `0.5187`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.3802`, XGBoost `0.5245`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.1324`, XGBoost `0.2705`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `17.0`, recent_utility `0`
