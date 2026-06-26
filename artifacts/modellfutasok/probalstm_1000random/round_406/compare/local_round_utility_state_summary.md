# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `12`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.344778 | 0.515082 | -0.170304 | 177 | 0 | 0.887006 | 0.372881 |
| active/recent utility | 177 | 1.000 | 0.344778 | 0.515082 | -0.170304 | 177 | 0 | 0.887006 | 0.372881 |
| strong utility action | 159 | 0.898 | 0.336276 | 0.515737 | -0.179461 | 159 | 0 | 0.874214 | 0.371069 |
| utility damage | 11 | 0.062 | 0.309523 | 0.520285 | -0.210762 | 11 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 159 | 0.898 | 0.336276 | 0.515737 | -0.179461 | 159 | 0 | 0.874214 | 0.371069 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 177 | 1.000 | 0.344778 | 0.515082 | -0.170304 | 177 | 0 | 0.887006 | 0.372881 |

## Active Smoke/Inferno Intervals

- `9.0s` - `88.0s`, rows `159`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.0`, LSTM `0.3294`, XGBoost `0.6678`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.3338`, XGBoost `0.6674`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.3463`, XGBoost `0.6683`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.3650`, XGBoost `0.6683`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.3765`, XGBoost `0.6730`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3800`, XGBoost `0.6723`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.3811`, XGBoost `0.6730`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.1418`, XGBoost `0.4291`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1661`, XGBoost `0.4505`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1494`, XGBoost `0.4291`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
