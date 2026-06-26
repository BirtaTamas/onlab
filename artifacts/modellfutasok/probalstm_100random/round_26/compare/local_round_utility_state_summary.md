# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `2`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.733565 | 0.750629 | -0.017064 | 49 | 147 | 0.969388 | 0.994898 |
| active/recent utility | 196 | 1.000 | 0.733565 | 0.750629 | -0.017064 | 49 | 147 | 0.969388 | 0.994898 |
| strong utility action | 160 | 0.816 | 0.767530 | 0.777672 | -0.010142 | 49 | 111 | 1.000000 | 0.993750 |
| utility damage | 13 | 0.066 | 0.772553 | 0.709967 | 0.062586 | 11 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 159 | 0.811 | 0.769181 | 0.778998 | -0.009817 | 49 | 110 | 1.000000 | 0.993711 |
| recent utility last 5s | 10 | 0.051 | 0.737450 | 0.752776 | -0.015326 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 196 | 1.000 | 0.733565 | 0.750629 | -0.017064 | 49 | 147 | 0.969388 | 0.994898 |

## Active Smoke/Inferno Intervals

- `12.0s` - `63.0s`, rows `103`
- `70.0s` - `97.5s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.8764`, XGBoost `0.7616`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.8616`, XGBoost `0.7561`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.8528`, XGBoost `0.7506`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.8461`, XGBoost `0.7502`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5615`, XGBoost `0.4663`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.8407`, XGBoost `0.7500`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.8391`, XGBoost `0.7488`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.8027`, XGBoost `0.8865`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.8044`, XGBoost `0.8865`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.8111`, XGBoost `0.8865`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
