# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `5`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.772202 | 0.762559 | 0.009643 | 140 | 90 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.772202 | 0.762559 | 0.009643 | 140 | 90 | 1.000000 | 1.000000 |
| strong utility action | 207 | 0.900 | 0.772280 | 0.764346 | 0.007934 | 124 | 83 | 1.000000 | 1.000000 |
| utility damage | 39 | 0.170 | 0.785920 | 0.788851 | -0.002931 | 16 | 23 | 1.000000 | 1.000000 |
| active smoke/inferno | 192 | 0.835 | 0.774425 | 0.767576 | 0.006849 | 114 | 78 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.708072 | 0.659567 | 0.048505 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.772202 | 0.762559 | 0.009643 | 140 | 90 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `12.0s` - `80.5s`, rows `138`
- `86.5s` - `113.0s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.5`, LSTM `0.7350`, XGBoost `0.8326`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.7563`, XGBoost `0.8459`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.7436`, XGBoost `0.8326`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.7481`, XGBoost `0.8338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.7490`, XGBoost `0.8326`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.7493`, XGBoost `0.8326`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.7315`, XGBoost `0.6593`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `93.5`, LSTM `0.7639`, XGBoost `0.8332`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.7793`, XGBoost `0.8459`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `0.5`, LSTM `0.7219`, XGBoost `0.6594`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
