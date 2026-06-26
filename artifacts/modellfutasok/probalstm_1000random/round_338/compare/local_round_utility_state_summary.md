# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `15`
- rows: `224`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.908490 | 0.955896 | -0.047406 | 0 | 224 | 1.000000 | 1.000000 |
| active/recent utility | 224 | 1.000 | 0.908490 | 0.955896 | -0.047406 | 0 | 224 | 1.000000 | 1.000000 |
| strong utility action | 104 | 0.464 | 0.861114 | 0.935799 | -0.074685 | 0 | 104 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.045 | 0.857268 | 0.942948 | -0.085681 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 104 | 0.464 | 0.861114 | 0.935799 | -0.074685 | 0 | 104 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.908490 | 0.955896 | -0.047406 | 0 | 224 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `59.5s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.5`, LSTM `0.7466`, XGBoost `0.9077`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.7626`, XGBoost `0.9082`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.7626`, XGBoost `0.9077`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7656`, XGBoost `0.9085`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7806`, XGBoost `0.9085`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7832`, XGBoost `0.9102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.7861`, XGBoost `0.9102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.7865`, XGBoost `0.9085`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.7913`, XGBoost `0.9102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.7905`, XGBoost `0.9085`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
