# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `3`
- rows: `159`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 159 | 1.000 | 0.007198 | 0.029815 | -0.022617 | 159 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 159 | 1.000 | 0.007198 | 0.029815 | -0.022617 | 159 | 0 | 1.000000 | 1.000000 |
| strong utility action | 94 | 0.591 | 0.009390 | 0.035877 | -0.026486 | 94 | 0 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.069 | 0.007185 | 0.046130 | -0.038945 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 92 | 0.579 | 0.009387 | 0.035292 | -0.025905 | 92 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 159 | 1.000 | 0.007198 | 0.029815 | -0.022617 | 159 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `30.0s`, rows `48`
- `36.5s` - `58.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.0093`, XGBoost `0.0692`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `112.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0132`, XGBoost `0.0656`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0096`, XGBoost `0.0589`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0146`, XGBoost `0.0637`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0154`, XGBoost `0.0644`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0156`, XGBoost `0.0643`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0129`, XGBoost `0.0612`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0112`, XGBoost `0.0594`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `71.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0157`, XGBoost `0.0638`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0139`, XGBoost `0.0620`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
