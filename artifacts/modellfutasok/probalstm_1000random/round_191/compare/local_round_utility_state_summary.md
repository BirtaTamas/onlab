# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `10`
- rows: `209`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 209 | 1.000 | 0.453543 | 0.472519 | -0.018976 | 141 | 68 | 0.406699 | 0.440191 |
| active/recent utility | 209 | 1.000 | 0.453543 | 0.472519 | -0.018976 | 141 | 68 | 0.406699 | 0.440191 |
| strong utility action | 169 | 0.809 | 0.523925 | 0.543070 | -0.019145 | 108 | 61 | 0.307692 | 0.349112 |
| utility damage | 10 | 0.048 | 0.726268 | 0.713923 | 0.012345 | 2 | 8 | 0.000000 | 0.000000 |
| active smoke/inferno | 159 | 0.761 | 0.508005 | 0.531299 | -0.023294 | 108 | 51 | 0.327044 | 0.371069 |
| recent utility last 5s | 10 | 0.048 | 0.777051 | 0.730231 | 0.046820 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 209 | 1.000 | 0.453543 | 0.472519 | -0.018976 | 141 | 68 | 0.406699 | 0.440191 |

## Active Smoke/Inferno Intervals

- `8.5s` - `87.5s`, rows `159`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.1062`, XGBoost `0.2857`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.1363`, XGBoost `0.2848`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.6146`, XGBoost `0.7595`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1339`, XGBoost `0.2774`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5232`, XGBoost `0.3848`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5171`, XGBoost `0.3921`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.1475`, XGBoost `0.2646`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5176`, XGBoost `0.4039`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0798`, XGBoost `0.1934`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1852`, XGBoost `0.0729`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
