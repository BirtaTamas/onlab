# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `3`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.691230 | 0.766391 | -0.075161 | 116 | 57 | 0.156069 | 0.017341 |
| active/recent utility | 173 | 1.000 | 0.691230 | 0.766391 | -0.075161 | 116 | 57 | 0.156069 | 0.017341 |
| strong utility action | 153 | 0.884 | 0.735065 | 0.782226 | -0.047161 | 99 | 54 | 0.065359 | 0.013072 |
| utility damage | 10 | 0.058 | 0.731849 | 0.757306 | -0.025457 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 142 | 0.821 | 0.723470 | 0.776870 | -0.053400 | 99 | 43 | 0.070423 | 0.014085 |
| recent utility last 5s | 12 | 0.069 | 0.881486 | 0.851316 | 0.030170 | 1 | 11 | 0.000000 | 0.000000 |
| flash effect present | 173 | 1.000 | 0.691230 | 0.766391 | -0.075161 | 116 | 57 | 0.156069 | 0.017341 |

## Active Smoke/Inferno Intervals

- `6.0s` - `11.0s`, rows `11`
- `12.5s` - `55.0s`, rows `86`
- `56.0s` - `78.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.4994`, XGBoost `0.8104`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5049`, XGBoost `0.8111`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5055`, XGBoost `0.8111`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5144`, XGBoost `0.8111`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5460`, XGBoost `0.8111`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3922`, XGBoost `0.6510`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5520`, XGBoost `0.8073`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5575`, XGBoost `0.8073`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5714`, XGBoost `0.8055`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.4204`, XGBoost `0.6503`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
