# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `14`
- rows: `116`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.083847 | 0.107226 | -0.023379 | 106 | 10 | 1.000000 | 1.000000 |
| active/recent utility | 116 | 1.000 | 0.083847 | 0.107226 | -0.023379 | 106 | 10 | 1.000000 | 1.000000 |
| strong utility action | 88 | 0.759 | 0.086696 | 0.108978 | -0.022282 | 78 | 10 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 74 | 0.638 | 0.034459 | 0.061958 | -0.027499 | 70 | 4 | 1.000000 | 1.000000 |
| recent utility last 5s | 14 | 0.121 | 0.362802 | 0.357512 | 0.005290 | 8 | 6 | 1.000000 | 1.000000 |
| flash effect present | 116 | 1.000 | 0.083847 | 0.107226 | -0.023379 | 106 | 10 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `17.0s`, rows `14`
- `21.5s` - `51.0s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.5`, LSTM `0.1793`, XGBoost `0.3463`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1818`, XGBoost `0.3452`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1881`, XGBoost `0.3463`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.1885`, XGBoost `0.3452`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1758`, XGBoost `0.3286`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1905`, XGBoost `0.3429`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1942`, XGBoost `0.3463`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1787`, XGBoost `0.3257`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1857`, XGBoost `0.3257`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2018`, XGBoost `0.3399`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
