# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `5`
- rows: `238`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.214067 | 0.214926 | -0.000859 | 162 | 76 | 0.894958 | 0.894958 |
| active/recent utility | 238 | 1.000 | 0.214067 | 0.214926 | -0.000859 | 162 | 76 | 0.894958 | 0.894958 |
| strong utility action | 148 | 0.622 | 0.228140 | 0.232406 | -0.004265 | 100 | 48 | 0.925676 | 0.925676 |
| utility damage | 19 | 0.080 | 0.291291 | 0.368820 | -0.077529 | 16 | 3 | 0.842105 | 0.842105 |
| active smoke/inferno | 148 | 0.622 | 0.228140 | 0.232406 | -0.004265 | 100 | 48 | 0.925676 | 0.925676 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 238 | 1.000 | 0.214067 | 0.214926 | -0.000859 | 162 | 76 | 0.894958 | 0.894958 |

## Active Smoke/Inferno Intervals

- `7.0s` - `38.5s`, rows `64`
- `51.0s` - `92.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.0`, LSTM `0.1624`, XGBoost `0.3532`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `71.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1741`, XGBoost `0.3538`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1830`, XGBoost `0.3506`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `51.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1503`, XGBoost `0.2977`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `83.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1588`, XGBoost `0.2996`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `83.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1680`, XGBoost `0.2991`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `83.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2270`, XGBoost `0.3475`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `33.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4872`, XGBoost `0.3713`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.4847`, XGBoost `0.3708`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.4820`, XGBoost `0.3713`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
