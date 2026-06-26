# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `9`
- rows: `237`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 237 | 1.000 | 0.499923 | 0.610237 | -0.110314 | 206 | 31 | 0.244726 | 0.181435 |
| active/recent utility | 237 | 1.000 | 0.499923 | 0.610237 | -0.110314 | 206 | 31 | 0.244726 | 0.181435 |
| strong utility action | 207 | 0.873 | 0.532910 | 0.638805 | -0.105895 | 189 | 18 | 0.198068 | 0.154589 |
| utility damage | 11 | 0.046 | 0.581355 | 0.625625 | -0.044270 | 10 | 1 | 0.000000 | 0.000000 |
| active smoke/inferno | 207 | 0.873 | 0.532910 | 0.638805 | -0.105895 | 189 | 18 | 0.198068 | 0.154589 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 237 | 1.000 | 0.499923 | 0.610237 | -0.110314 | 206 | 31 | 0.244726 | 0.181435 |

## Active Smoke/Inferno Intervals

- `6.5s` - `105.5s`, rows `199`
- `114.5s` - `118.0s`, rows `8`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `115.0`, LSTM `0.2121`, XGBoost `0.7567`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `115.5`, LSTM `0.2203`, XGBoost `0.7567`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `116.0`, LSTM `0.2310`, XGBoost `0.7533`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.1034`, XGBoost `0.6118`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `116.5`, LSTM `0.2718`, XGBoost `0.7489`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `117.5`, LSTM `0.3154`, XGBoost `0.7489`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `117.0`, LSTM `0.3157`, XGBoost `0.7489`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `118.0`, LSTM `0.3266`, XGBoost `0.7471`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.0785`, XGBoost `0.4087`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.0724`, XGBoost `0.3784`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
