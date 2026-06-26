# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `8`
- rows: `134`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.783473 | 0.821310 | -0.037837 | 20 | 114 | 1.000000 | 1.000000 |
| active/recent utility | 134 | 1.000 | 0.783473 | 0.821310 | -0.037837 | 20 | 114 | 1.000000 | 1.000000 |
| strong utility action | 95 | 0.709 | 0.737770 | 0.776560 | -0.038790 | 18 | 77 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.157 | 0.686766 | 0.751050 | -0.064284 | 0 | 21 | 1.000000 | 1.000000 |
| active smoke/inferno | 85 | 0.634 | 0.740917 | 0.788109 | -0.047191 | 9 | 76 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.075 | 0.711017 | 0.678402 | 0.032615 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 134 | 1.000 | 0.783473 | 0.821310 | -0.037837 | 20 | 114 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `50.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.7055`, XGBoost `0.8466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7056`, XGBoost `0.8465`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7152`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5033`, XGBoost `0.6344`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7259`, XGBoost `0.8464`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `22.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7307`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.5121`, XGBoost `0.6264`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7351`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7383`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5255`, XGBoost `0.6334`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `7.0`, recent_utility `0`
