# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `10`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.165491 | 0.191387 | -0.025896 | 112 | 30 | 0.816901 | 0.971831 |
| active/recent utility | 142 | 1.000 | 0.165491 | 0.191387 | -0.025896 | 112 | 30 | 0.816901 | 0.971831 |
| strong utility action | 128 | 0.901 | 0.158532 | 0.186698 | -0.028166 | 104 | 24 | 0.843750 | 0.968750 |
| utility damage | 10 | 0.070 | 0.345276 | 0.431643 | -0.086368 | 9 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.845 | 0.135156 | 0.166057 | -0.030901 | 104 | 16 | 0.900000 | 0.975000 |
| recent utility last 5s | 10 | 0.070 | 0.511714 | 0.493876 | 0.017837 | 0 | 10 | 0.000000 | 0.900000 |
| flash effect present | 142 | 1.000 | 0.165491 | 0.191387 | -0.025896 | 112 | 30 | 0.816901 | 0.971831 |

## Active Smoke/Inferno Intervals

- `7.0s` - `58.5s`, rows `104`
- `63.0s` - `70.5s`, rows `16`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.2878`, XGBoost `0.4394`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.2787`, XGBoost `0.4283`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.2811`, XGBoost `0.4287`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0552`, XGBoost `0.1956`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0486`, XGBoost `0.1869`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.2964`, XGBoost `0.4307`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2993`, XGBoost `0.4307`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3117`, XGBoost `0.4307`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3178`, XGBoost `0.4368`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.3165`, XGBoost `0.4347`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
