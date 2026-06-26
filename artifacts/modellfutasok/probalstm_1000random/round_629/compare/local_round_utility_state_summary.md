# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `17`
- rows: `164`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.848640 | 0.841708 | 0.006932 | 65 | 99 | 1.000000 | 1.000000 |
| active/recent utility | 164 | 1.000 | 0.848640 | 0.841708 | 0.006932 | 65 | 99 | 1.000000 | 1.000000 |
| strong utility action | 140 | 0.854 | 0.872566 | 0.870860 | 0.001706 | 46 | 94 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.061 | 0.969974 | 0.977930 | -0.007956 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 140 | 0.854 | 0.872566 | 0.870860 | 0.001706 | 46 | 94 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 164 | 1.000 | 0.848640 | 0.841708 | 0.006932 | 65 | 99 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `74.5s`, rows `134`
- `79.0s` - `81.5s`, rows `6`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `80.5`, LSTM `0.5670`, XGBoost `0.6890`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.5727`, XGBoost `0.6900`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.6213`, XGBoost `0.7250`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5801`, XGBoost `0.6833`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.7107`, XGBoost `0.6173`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.7082`, XGBoost `0.6164`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.7072`, XGBoost `0.6173`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.7054`, XGBoost `0.6164`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.7328`, XGBoost `0.6447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6030`, XGBoost `0.6912`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
