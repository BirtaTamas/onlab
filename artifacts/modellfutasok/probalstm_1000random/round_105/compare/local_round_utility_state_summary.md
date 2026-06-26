# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `11`
- rows: `249`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 249 | 1.000 | 0.349849 | 0.409617 | -0.059768 | 236 | 13 | 0.510040 | 0.449799 |
| active/recent utility | 249 | 1.000 | 0.349849 | 0.409617 | -0.059768 | 236 | 13 | 0.510040 | 0.449799 |
| strong utility action | 140 | 0.562 | 0.391369 | 0.441186 | -0.049817 | 127 | 13 | 0.435714 | 0.421429 |
| utility damage | 10 | 0.040 | 0.562224 | 0.638587 | -0.076363 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 140 | 0.562 | 0.391369 | 0.441186 | -0.049817 | 127 | 13 | 0.435714 | 0.421429 |
| recent utility last 5s | 10 | 0.040 | 0.074800 | 0.102968 | -0.028168 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 249 | 1.000 | 0.349849 | 0.409617 | -0.059768 | 236 | 13 | 0.510040 | 0.449799 |

## Active Smoke/Inferno Intervals

- `6.5s` - `33.5s`, rows `55`
- `55.5s` - `97.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.6235`, XGBoost `0.7947`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6249`, XGBoost `0.7910`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6287`, XGBoost `0.7947`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4958`, XGBoost `0.3358`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.4906`, XGBoost `0.3333`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6376`, XGBoost `0.7947`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6377`, XGBoost `0.7947`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4947`, XGBoost `0.3426`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6360`, XGBoost `0.7874`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6411`, XGBoost `0.7910`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
