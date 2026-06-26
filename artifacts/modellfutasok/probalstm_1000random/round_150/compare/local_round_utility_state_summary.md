# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `20`
- rows: `194`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.793996 | 0.863866 | -0.069870 | 1 | 193 | 1.000000 | 1.000000 |
| active/recent utility | 194 | 1.000 | 0.793996 | 0.863866 | -0.069870 | 1 | 193 | 1.000000 | 1.000000 |
| strong utility action | 137 | 0.706 | 0.793955 | 0.859297 | -0.065343 | 1 | 136 | 1.000000 | 1.000000 |
| utility damage | 38 | 0.196 | 0.688750 | 0.785514 | -0.096764 | 0 | 38 | 1.000000 | 1.000000 |
| active smoke/inferno | 127 | 0.655 | 0.801638 | 0.870218 | -0.068580 | 1 | 126 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.052 | 0.696381 | 0.720610 | -0.024229 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 194 | 1.000 | 0.793996 | 0.863866 | -0.069870 | 1 | 193 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `71.5s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.5`, LSTM `0.6987`, XGBoost `0.9281`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7334`, XGBoost `0.9282`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6300`, XGBoost `0.8136`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7476`, XGBoost `0.9279`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `103.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7524`, XGBoost `0.9281`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6005`, XGBoost `0.7743`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6517`, XGBoost `0.8176`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6532`, XGBoost `0.8176`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7697`, XGBoost `0.9281`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `103.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6602`, XGBoost `0.8108`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
