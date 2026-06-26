# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `19`
- rows: `104`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 104 | 1.000 | 0.498126 | 0.442009 | 0.056117 | 13 | 91 | 0.336538 | 0.471154 |
| active/recent utility | 104 | 1.000 | 0.498126 | 0.442009 | 0.056117 | 13 | 91 | 0.336538 | 0.471154 |
| strong utility action | 85 | 0.817 | 0.472241 | 0.413936 | 0.058305 | 13 | 72 | 0.411765 | 0.576471 |
| utility damage | 21 | 0.202 | 0.500596 | 0.471265 | 0.029331 | 3 | 18 | 0.380952 | 0.380952 |
| active smoke/inferno | 85 | 0.817 | 0.472241 | 0.413936 | 0.058305 | 13 | 72 | 0.411765 | 0.576471 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 104 | 1.000 | 0.498126 | 0.442009 | 0.056117 | 13 | 91 | 0.336538 | 0.471154 |

## Active Smoke/Inferno Intervals

- `9.5s` - `51.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `46.5`, LSTM `0.3395`, XGBoost `0.0749`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2727`, XGBoost `0.0749`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5602`, XGBoost `0.3872`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5561`, XGBoost `0.3872`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5445`, XGBoost `0.3872`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5437`, XGBoost `0.3872`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5391`, XGBoost `0.3864`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5325`, XGBoost `0.3864`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5311`, XGBoost `0.3864`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.2193`, XGBoost `0.0751`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
