# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `5`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.273384 | 0.329972 | -0.056587 | 119 | 23 | 0.704225 | 0.690141 |
| active/recent utility | 142 | 1.000 | 0.273384 | 0.329972 | -0.056587 | 119 | 23 | 0.704225 | 0.690141 |
| strong utility action | 75 | 0.528 | 0.310669 | 0.397763 | -0.087094 | 68 | 7 | 0.666667 | 0.640000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 75 | 0.528 | 0.310669 | 0.397763 | -0.087094 | 68 | 7 | 0.666667 | 0.640000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.273384 | 0.329972 | -0.056587 | 119 | 23 | 0.704225 | 0.690141 |

## Active Smoke/Inferno Intervals

- `8.5s` - `45.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.2052`, XGBoost `0.4479`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1997`, XGBoost `0.4385`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1173`, XGBoost `0.3522`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1325`, XGBoost `0.3650`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.1330`, XGBoost `0.3637`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2082`, XGBoost `0.4377`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.1409`, XGBoost `0.3650`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0395`, XGBoost `0.2504`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3395`, XGBoost `0.5470`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0545`, XGBoost `0.2556`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
