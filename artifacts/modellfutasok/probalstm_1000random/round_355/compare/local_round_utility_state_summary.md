# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `12`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.222122 | 0.205261 | 0.016861 | 122 | 75 | 0.675127 | 0.675127 |
| active/recent utility | 197 | 1.000 | 0.222122 | 0.205261 | 0.016861 | 122 | 75 | 0.675127 | 0.675127 |
| strong utility action | 163 | 0.827 | 0.203020 | 0.182660 | 0.020360 | 102 | 61 | 0.717791 | 0.717791 |
| utility damage | 10 | 0.051 | 0.353213 | 0.314071 | 0.039141 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 159 | 0.807 | 0.197453 | 0.177168 | 0.020285 | 100 | 59 | 0.710692 | 0.710692 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.222122 | 0.205261 | 0.016861 | 122 | 75 | 0.675127 | 0.675127 |

## Active Smoke/Inferno Intervals

- `9.0s` - `32.0s`, rows `47`
- `36.0s` - `91.5s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.6057`, XGBoost `0.5045`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `5.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4892`, XGBoost `0.3881`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `41.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6055`, XGBoost `0.5079`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.2441`, XGBoost `0.1513`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4926`, XGBoost `0.3999`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `41.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.5983`, XGBoost `0.5059`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `5.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5988`, XGBoost `0.5084`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.5945`, XGBoost `0.5053`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `5.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.5938`, XGBoost `0.5053`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `5.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4847`, XGBoost `0.3975`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `41.0`, recent_utility `0`
