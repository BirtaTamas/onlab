# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `2`
- rows: `268`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 268 | 1.000 | 0.062393 | 0.073040 | -0.010647 | 158 | 110 | 1.000000 | 1.000000 |
| active/recent utility | 268 | 1.000 | 0.062393 | 0.073040 | -0.010647 | 158 | 110 | 1.000000 | 1.000000 |
| strong utility action | 123 | 0.459 | 0.107218 | 0.122999 | -0.015781 | 58 | 65 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 113 | 0.422 | 0.099176 | 0.100257 | -0.001081 | 48 | 65 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.037 | 0.198088 | 0.379985 | -0.181898 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 268 | 1.000 | 0.062393 | 0.073040 | -0.010647 | 158 | 110 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `43.0s`, rows `69`
- `64.0s` - `85.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.0`, LSTM `0.1605`, XGBoost `0.3779`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.1756`, XGBoost `0.3827`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.1769`, XGBoost `0.3779`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.1823`, XGBoost `0.3827`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1944`, XGBoost `0.3779`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.1979`, XGBoost `0.3779`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.2046`, XGBoost `0.3827`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `9.0`, LSTM `0.2095`, XGBoost `0.3736`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2123`, XGBoost `0.3759`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.2238`, XGBoost `0.3790`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
