# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `18`
- rows: `200`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.132938 | 0.143974 | -0.011036 | 158 | 42 | 1.000000 | 1.000000 |
| active/recent utility | 200 | 1.000 | 0.132938 | 0.143974 | -0.011036 | 158 | 42 | 1.000000 | 1.000000 |
| strong utility action | 132 | 0.660 | 0.136924 | 0.156096 | -0.019172 | 113 | 19 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 132 | 0.660 | 0.136924 | 0.156096 | -0.019172 | 113 | 19 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 200 | 1.000 | 0.132938 | 0.143974 | -0.011036 | 158 | 42 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `33.5s`, rows `49`
- `38.5s` - `79.5s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.2342`, XGBoost `0.3490`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.2350`, XGBoost `0.3490`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.2437`, XGBoost `0.3490`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4532`, XGBoost `0.3512`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.2482`, XGBoost `0.3490`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.2552`, XGBoost `0.3490`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4384`, XGBoost `0.3508`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4305`, XGBoost `0.3530`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2706`, XGBoost `0.3475`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4224`, XGBoost `0.3495`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
