# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `6`
- rows: `242`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 242 | 1.000 | 0.122014 | 0.188328 | -0.066314 | 242 | 0 | 0.979339 | 0.772727 |
| active/recent utility | 242 | 1.000 | 0.122014 | 0.188328 | -0.066314 | 242 | 0 | 0.979339 | 0.772727 |
| strong utility action | 177 | 0.731 | 0.134042 | 0.217007 | -0.082965 | 177 | 0 | 0.971751 | 0.757062 |
| utility damage | 35 | 0.145 | 0.333506 | 0.442586 | -0.109081 | 35 | 0 | 0.942857 | 0.285714 |
| active smoke/inferno | 177 | 0.731 | 0.134042 | 0.217007 | -0.082965 | 177 | 0 | 0.971751 | 0.757062 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 242 | 1.000 | 0.122014 | 0.188328 | -0.066314 | 242 | 0 | 0.979339 | 0.772727 |

## Active Smoke/Inferno Intervals

- `6.0s` - `66.5s`, rows `122`
- `74.0s` - `95.5s`, rows `44`
- `106.5s` - `111.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.0955`, XGBoost `0.3626`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1044`, XGBoost `0.3626`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1075`, XGBoost `0.3626`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1024`, XGBoost `0.3569`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1164`, XGBoost `0.3626`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1112`, XGBoost `0.3572`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1124`, XGBoost `0.3572`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1213`, XGBoost `0.3626`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1159`, XGBoost `0.3569`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.1169`, XGBoost `0.3577`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
