# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `17`
- rows: `186`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.813281 | 0.855255 | -0.041974 | 32 | 154 | 1.000000 | 1.000000 |
| active/recent utility | 186 | 1.000 | 0.813281 | 0.855255 | -0.041974 | 32 | 154 | 1.000000 | 1.000000 |
| strong utility action | 102 | 0.548 | 0.757968 | 0.826082 | -0.068113 | 16 | 86 | 1.000000 | 1.000000 |
| utility damage | 15 | 0.081 | 0.748879 | 0.850730 | -0.101850 | 0 | 15 | 1.000000 | 1.000000 |
| active smoke/inferno | 102 | 0.548 | 0.757968 | 0.826082 | -0.068113 | 16 | 86 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 186 | 1.000 | 0.813281 | 0.855255 | -0.041974 | 32 | 154 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.5s`, rows `88`
- `60.0s` - `66.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.0`, LSTM `0.6544`, XGBoost `0.8512`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.6600`, XGBoost `0.8512`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.6696`, XGBoost `0.8540`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6686`, XGBoost `0.8526`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.6731`, XGBoost `0.8522`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.6333`, XGBoost `0.7934`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6339`, XGBoost `0.7934`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.7003`, XGBoost `0.8526`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.7021`, XGBoost `0.8526`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6385`, XGBoost `0.7841`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
