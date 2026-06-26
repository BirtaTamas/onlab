# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `9`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.567181 | 0.617319 | -0.050138 | 47 | 140 | 0.780749 | 0.887701 |
| active/recent utility | 187 | 1.000 | 0.567181 | 0.617319 | -0.050138 | 47 | 140 | 0.780749 | 0.887701 |
| strong utility action | 156 | 0.834 | 0.559310 | 0.606603 | -0.047294 | 46 | 110 | 0.814103 | 0.865385 |
| utility damage | 20 | 0.107 | 0.621747 | 0.652065 | -0.030318 | 9 | 11 | 0.950000 | 0.900000 |
| active smoke/inferno | 156 | 0.834 | 0.559310 | 0.606603 | -0.047294 | 46 | 110 | 0.814103 | 0.865385 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 187 | 1.000 | 0.567181 | 0.617319 | -0.050138 | 47 | 140 | 0.780749 | 0.887701 |

## Active Smoke/Inferno Intervals

- `6.5s` - `84.0s`, rows `156`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.5`, LSTM `0.5226`, XGBoost `0.6796`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5300`, XGBoost `0.6796`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5333`, XGBoost `0.6784`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5359`, XGBoost `0.6784`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5300`, XGBoost `0.6723`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.5917`, XGBoost `0.7337`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.3775`, XGBoost `0.5189`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5316`, XGBoost `0.6723`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5932`, XGBoost `0.7322`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5426`, XGBoost `0.6796`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
