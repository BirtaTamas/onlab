# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `14`
- rows: `225`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.272516 | 0.386754 | -0.114238 | 209 | 16 | 0.955556 | 0.897778 |
| active/recent utility | 225 | 1.000 | 0.272516 | 0.386754 | -0.114238 | 209 | 16 | 0.955556 | 0.897778 |
| strong utility action | 204 | 0.907 | 0.283449 | 0.393489 | -0.110040 | 188 | 16 | 0.950980 | 0.887255 |
| utility damage | 10 | 0.044 | 0.250542 | 0.431357 | -0.180815 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 204 | 0.907 | 0.283449 | 0.393489 | -0.110040 | 188 | 16 | 0.950980 | 0.887255 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 225 | 1.000 | 0.272516 | 0.386754 | -0.114238 | 209 | 16 | 0.955556 | 0.897778 |

## Active Smoke/Inferno Intervals

- `6.0s` - `63.0s`, rows `115`
- `65.0s` - `109.0s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.0`, LSTM `0.1444`, XGBoost `0.4154`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1576`, XGBoost `0.4119`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1597`, XGBoost `0.4119`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1601`, XGBoost `0.4119`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.1635`, XGBoost `0.4133`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1694`, XGBoost `0.4147`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1697`, XGBoost `0.4119`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `38.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1787`, XGBoost `0.4197`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1767`, XGBoost `0.4133`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `38.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1736`, XGBoost `0.4062`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
