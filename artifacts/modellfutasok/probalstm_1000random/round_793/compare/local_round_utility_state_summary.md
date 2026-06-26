# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `3`
- rows: `191`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.415658 | 0.360343 | 0.055315 | 49 | 142 | 0.523560 | 0.952880 |
| active/recent utility | 191 | 1.000 | 0.415658 | 0.360343 | 0.055315 | 49 | 142 | 0.523560 | 0.952880 |
| strong utility action | 156 | 0.817 | 0.427284 | 0.361599 | 0.065685 | 39 | 117 | 0.493590 | 0.942308 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 156 | 0.817 | 0.427284 | 0.361599 | 0.065685 | 39 | 117 | 0.493590 | 0.942308 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 191 | 1.000 | 0.415658 | 0.360343 | 0.055315 | 49 | 142 | 0.523560 | 0.952880 |

## Active Smoke/Inferno Intervals

- `8.0s` - `63.5s`, rows `112`
- `71.0s` - `92.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.0`, LSTM `0.3168`, XGBoost `0.5197`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.4504`, XGBoost `0.2567`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.4425`, XGBoost `0.2567`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6199`, XGBoost `0.4357`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6177`, XGBoost `0.4365`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6160`, XGBoost `0.4357`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.4336`, XGBoost `0.2567`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6079`, XGBoost `0.4357`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6077`, XGBoost `0.4357`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6084`, XGBoost `0.4377`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
