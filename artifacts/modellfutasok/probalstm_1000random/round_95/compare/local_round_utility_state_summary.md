# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `16`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.183088 | 0.214870 | -0.031782 | 194 | 3 | 0.868020 | 0.644670 |
| active/recent utility | 197 | 1.000 | 0.183088 | 0.214870 | -0.031782 | 194 | 3 | 0.868020 | 0.644670 |
| strong utility action | 129 | 0.655 | 0.266936 | 0.308982 | -0.042047 | 126 | 3 | 0.798450 | 0.480620 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 117 | 0.594 | 0.244864 | 0.285064 | -0.040199 | 114 | 3 | 0.786325 | 0.529915 |
| recent utility last 5s | 12 | 0.061 | 0.482131 | 0.542190 | -0.060059 | 12 | 0 | 0.916667 | 0.000000 |
| flash effect present | 197 | 1.000 | 0.183088 | 0.214870 | -0.031782 | 194 | 3 | 0.868020 | 0.644670 |

## Active Smoke/Inferno Intervals

- `7.5s` - `65.5s`, rows `117`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.3587`, XGBoost `0.5420`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.3812`, XGBoost `0.5420`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.3221`, XGBoost `0.1681`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.3963`, XGBoost `0.5433`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4010`, XGBoost `0.5448`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.4021`, XGBoost `0.5448`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.4038`, XGBoost `0.5439`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4069`, XGBoost `0.5420`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4159`, XGBoost `0.5420`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0721`, XGBoost `0.1849`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
