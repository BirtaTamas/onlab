# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `15`
- rows: `144`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.099617 | 0.241133 | -0.141515 | 144 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 144 | 1.000 | 0.099617 | 0.241133 | -0.141515 | 144 | 0 | 1.000000 | 1.000000 |
| strong utility action | 75 | 0.521 | 0.082839 | 0.229764 | -0.146925 | 75 | 0 | 1.000000 | 1.000000 |
| utility damage | 17 | 0.118 | 0.083910 | 0.237580 | -0.153670 | 17 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 58 | 0.403 | 0.082525 | 0.227473 | -0.144948 | 58 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.099617 | 0.241133 | -0.141515 | 144 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `16.0s`, rows `14`
- `35.0s` - `56.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.5`, LSTM `0.1411`, XGBoost `0.4316`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1454`, XGBoost `0.4316`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1671`, XGBoost `0.4323`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1810`, XGBoost `0.4333`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0802`, XGBoost `0.3310`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `149.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2131`, XGBoost `0.4323`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.1310`, XGBoost `0.3310`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `149.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1410`, XGBoost `0.3409`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `90.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1427`, XGBoost `0.3310`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `90.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1004`, XGBoost `0.2840`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `86.0`, recent_utility `0`
