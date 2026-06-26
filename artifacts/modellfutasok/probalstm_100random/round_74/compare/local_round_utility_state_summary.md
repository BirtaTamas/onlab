# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `5`
- rows: `253`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 253 | 1.000 | 0.161359 | 0.208582 | -0.047223 | 213 | 40 | 1.000000 | 1.000000 |
| active/recent utility | 253 | 1.000 | 0.161359 | 0.208582 | -0.047223 | 213 | 40 | 1.000000 | 1.000000 |
| strong utility action | 169 | 0.668 | 0.187383 | 0.242242 | -0.054859 | 142 | 27 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 169 | 0.668 | 0.187383 | 0.242242 | -0.054859 | 142 | 27 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 253 | 1.000 | 0.161359 | 0.208582 | -0.047223 | 213 | 40 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `35.5s`, rows `55`
- `44.5s` - `94.0s`, rows `100`
- `106.0s` - `112.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.1428`, XGBoost `0.3418`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1419`, XGBoost `0.3362`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1475`, XGBoost `0.3370`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1497`, XGBoost `0.3370`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1610`, XGBoost `0.3479`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1505`, XGBoost `0.3371`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1506`, XGBoost `0.3371`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1542`, XGBoost `0.3384`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1627`, XGBoost `0.3462`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.1691`, XGBoost `0.3479`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
