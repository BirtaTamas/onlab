# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `1`
- rows: `122`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 122 | 1.000 | 0.509251 | 0.631132 | -0.121881 | 18 | 104 | 0.327869 | 0.442623 |
| active/recent utility | 122 | 1.000 | 0.509251 | 0.631132 | -0.121881 | 18 | 104 | 0.327869 | 0.442623 |
| strong utility action | 56 | 0.459 | 0.353066 | 0.557979 | -0.204914 | 0 | 56 | 0.017857 | 0.392857 |
| utility damage | 10 | 0.082 | 0.280404 | 0.455674 | -0.175270 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 45 | 0.369 | 0.329521 | 0.573341 | -0.243820 | 0 | 45 | 0.022222 | 0.488889 |
| recent utility last 5s | 11 | 0.090 | 0.449385 | 0.495137 | -0.045752 | 0 | 11 | 0.000000 | 0.000000 |
| flash effect present | 122 | 1.000 | 0.509251 | 0.631132 | -0.121881 | 18 | 104 | 0.327869 | 0.442623 |

## Active Smoke/Inferno Intervals

- `19.5s` - `41.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.3557`, XGBoost `0.7387`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.3616`, XGBoost `0.7366`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3651`, XGBoost `0.7387`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3575`, XGBoost `0.7272`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3247`, XGBoost `0.6555`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3312`, XGBoost `0.6546`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4111`, XGBoost `0.7342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4216`, XGBoost `0.7357`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4256`, XGBoost `0.7342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3508`, XGBoost `0.6546`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
