# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `13`
- rows: `129`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 129 | 1.000 | 0.211054 | 0.213502 | -0.002447 | 81 | 48 | 0.813953 | 0.914729 |
| active/recent utility | 129 | 1.000 | 0.211054 | 0.213502 | -0.002447 | 81 | 48 | 0.813953 | 0.914729 |
| strong utility action | 68 | 0.527 | 0.248138 | 0.253610 | -0.005472 | 41 | 27 | 0.926471 | 0.852941 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 68 | 0.527 | 0.248138 | 0.253610 | -0.005472 | 41 | 27 | 0.926471 | 0.852941 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 129 | 1.000 | 0.211054 | 0.213502 | -0.002447 | 81 | 48 | 0.813953 | 0.914729 |

## Active Smoke/Inferno Intervals

- `10.0s` - `43.5s`, rows `68`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.5`, LSTM `0.5508`, XGBoost `0.7329`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.3495`, XGBoost `0.2486`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0682`, XGBoost `0.1652`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0731`, XGBoost `0.1652`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.3957`, XGBoost `0.4834`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0826`, XGBoost `0.1693`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0725`, XGBoost `0.1515`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.3986`, XGBoost `0.4707`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0877`, XGBoost `0.1523`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2928`, XGBoost `0.2400`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
