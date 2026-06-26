# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `10`
- rows: `205`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.803722 | 0.794224 | 0.009497 | 78 | 127 | 1.000000 | 0.980488 |
| active/recent utility | 205 | 1.000 | 0.803722 | 0.794224 | 0.009497 | 78 | 127 | 1.000000 | 0.980488 |
| strong utility action | 167 | 0.815 | 0.816054 | 0.813455 | 0.002599 | 56 | 111 | 1.000000 | 0.982036 |
| utility damage | 10 | 0.049 | 0.960559 | 0.990422 | -0.029863 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.766 | 0.806850 | 0.802183 | 0.004667 | 56 | 101 | 1.000000 | 0.980892 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.803722 | 0.794224 | 0.009497 | 78 | 127 | 1.000000 | 0.980488 |

## Active Smoke/Inferno Intervals

- `11.0s` - `81.0s`, rows `141`
- `83.0s` - `90.5s`, rows `16`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.6427`, XGBoost `0.5203`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6111`, XGBoost `0.4895`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6275`, XGBoost `0.5095`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6070`, XGBoost `0.4895`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6399`, XGBoost `0.5251`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6080`, XGBoost `0.4941`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6341`, XGBoost `0.5203`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6228`, XGBoost `0.5090`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6337`, XGBoost `0.5203`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6325`, XGBoost `0.5199`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
